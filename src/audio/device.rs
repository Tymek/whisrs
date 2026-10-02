//! Input device selection for `[audio] device`.
//!
//! `"default"` keeps the ALSA host's default input. A named device is looked
//! up among PulseAudio sources first (PipeWire answers the same protocol),
//! then among ALSA PCMs, and an unknown name falls back to the default.
//! The PulseAudio connection is cached for the process.
//! Playback never goes through here: it always uses [`alsa_host`].

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{BufferSize, DeviceId, ErrorKind, HostId};
use tracing::{debug, warn};

/// Buffer for named PulseAudio sources: 100 ms at 16 kHz. With
/// `BufferSize::Default` the PulseAudio host delivers 2 s fragments and
/// `play()` blocks for about as long.
const PULSE_INPUT_FRAMES: u32 = 1600;

/// Period for the ALSA default input: 50 ms at 16 kHz. cpal 0.18 opens an
/// ALSA stream with 2 periods, so this is a 100 ms ring, the same buffer
/// cpal 0.15 asked for (25 ms periods, 100 ms buffer). Left at
/// `BufferSize::Default`, cpal 0.18 takes PipeWire-ALSA's period (128 ms
/// here), which drops up to that much audio at stop and updates the level
/// meter at 8 Hz. Capture falls back to `Default` if the device rejects it.
pub const ALSA_DEFAULT_INPUT_FRAMES: u32 = 800;

/// Bound on creating a PulseAudio record stream. Without it cpal waits
/// forever on a server that accepted the connection but stopped answering.
pub const PULSE_BUILD_TIMEOUT: Duration = Duration::from_secs(3);

/// Which cpal host a candidate device belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceHost {
    PulseAudio,
    Alsa,
}

/// An input device as listed by a host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputDeviceInfo {
    pub host: DeviceHost,
    /// The bare device name, the value to put in `[audio] device`.
    pub id: String,
    /// Human-readable description.
    pub description: String,
}

impl InputDeviceInfo {
    /// A PulseAudio monitor source, i.e. what an output device is playing.
    pub fn is_monitor(&self) -> bool {
        self.host == DeviceHost::PulseAudio && self.id.ends_with(".monitor")
    }
}

/// How the configured device name was resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputSource {
    /// `"default"`: the ALSA default input.
    Default,
    /// A named device on the given host.
    Named(DeviceHost, String),
    /// The name matched nothing; using the ALSA default input.
    Fallback,
}

impl std::fmt::Display for InputSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Default => write!(f, "system default (ALSA)"),
            Self::Named(DeviceHost::PulseAudio, id) => write!(f, "PulseAudio source {id}"),
            Self::Named(DeviceHost::Alsa, id) => write!(f, "ALSA device {id}"),
            Self::Fallback => write!(f, "system default (ALSA, fallback)"),
        }
    }
}

/// A resolved input device plus the buffer size to open it with.
pub struct ResolvedInput {
    pub device: cpal::Device,
    pub buffer_size: BufferSize,
    /// Timeout for `build_input_stream`: set for PulseAudio sources, `None`
    /// for ALSA (where cpal uses it as a poll interval instead).
    pub build_timeout: Option<Duration>,
    pub source: InputSource,
    /// The device came from the cached PulseAudio connection. If opening it
    /// fails with a host-level error ([`is_host_level_error`]), call
    /// [`invalidate_pulse_host`] and resolve again: the server may have
    /// restarted under the cached connection.
    pub cached_pulse_host: bool,
}

/// True when the configured name means "use the system default".
pub fn is_default_name(name: &str) -> bool {
    let name = name.trim();
    name.is_empty() || name == "default"
}

/// Pick the candidate matching `name`: PulseAudio before ALSA, and within a
/// host an exact id before a description match.
pub fn match_device(name: &str, candidates: &[InputDeviceInfo]) -> Option<usize> {
    let name = name.trim();
    for host in [DeviceHost::PulseAudio, DeviceHost::Alsa] {
        let on_host = || {
            candidates
                .iter()
                .enumerate()
                .filter(|(_, c)| c.host == host)
        };
        if let Some((i, _)) = on_host().find(|(_, c)| c.id == name) {
            return Some(i);
        }
        if let Some((i, _)) = on_host().find(|(_, c)| c.description == name) {
            return Some(i);
        }
    }
    None
}

/// The ALSA host, used for the default input and for all playback.
pub fn alsa_host() -> anyhow::Result<cpal::Host> {
    cpal::host_from_id(HostId::Alsa)
        .map_err(|e| anyhow::anyhow!("ALSA audio host unavailable: {e}"))
}

/// Who is looking up devices. A diagnostic lookup (the startup check, device
/// lists) reads the PulseAudio backoff but never starts it, and does not
/// record an unknown-name warning: at login it can run before the sound
/// server is up, and a capture in the next [`PULSE_RETRY_AFTER`] must still
/// try PulseAudio and warn on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lookup {
    Capture,
    Diagnostic,
}

/// How long a failed PulseAudio connection is remembered before retrying.
/// A hung socket costs cpal's 2 s init timeout (and a leaked thread) per try.
const PULSE_RETRY_AFTER: Duration = Duration::from_secs(30);

/// The process-wide PulseAudio connection, so a named device does not cost a
/// handshake on every recording.
struct PulseSlot {
    host: Option<Arc<cpal::Host>>,
    failed_at: Option<Instant>,
}

static PULSE: Mutex<PulseSlot> = Mutex::new(PulseSlot {
    host: None,
    failed_at: None,
});

fn pulse_slot() -> std::sync::MutexGuard<'static, PulseSlot> {
    PULSE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether a PulseAudio connection may be attempted, given the last failure.
fn pulse_retry_due(failed_at: Option<Instant>, now: Instant) -> bool {
    failed_at.is_none_or(|t| now.saturating_duration_since(t) >= PULSE_RETRY_AFTER)
}

/// Record a failed PulseAudio connection: drop the cached host, and back
/// off unless the lookup is diagnostic.
fn record_pulse_failure(slot: &mut PulseSlot, now: Instant, lookup: Lookup) {
    slot.host = None;
    if lookup == Lookup::Capture {
        slot.failed_at = Some(now);
    }
}

/// The cached PulseAudio host, connecting if there is none. The flag is true
/// when the host came from the cache. `None` while a recent failure is
/// remembered (see [`PULSE_RETRY_AFTER`]).
fn pulse_host(lookup: Lookup) -> Option<(Arc<cpal::Host>, bool)> {
    if !cpal::available_hosts().contains(&HostId::PulseAudio) {
        return None;
    }
    let mut slot = pulse_slot();
    if let Some(host) = &slot.host {
        return Some((Arc::clone(host), true));
    }
    if !pulse_retry_due(slot.failed_at, Instant::now()) {
        return None;
    }
    match cpal::host_from_id(HostId::PulseAudio) {
        Ok(host) => {
            let host = Arc::new(host);
            slot.host = Some(Arc::clone(&host));
            slot.failed_at = None;
            Some((host, false))
        }
        Err(e) => {
            match lookup {
                Lookup::Capture => warn!(
                    "PulseAudio host unavailable ({e}); using ALSA devices only, \
                     retrying in {} s",
                    PULSE_RETRY_AFTER.as_secs()
                ),
                Lookup::Diagnostic => warn!(
                    "PulseAudio host unavailable ({e}); listing ALSA devices only, \
                     recording will try again"
                ),
            }
            record_pulse_failure(&mut slot, Instant::now(), lookup);
            None
        }
    }
}

/// Whether a cpal error kind means the audio server itself is gone (it died
/// or restarted), as opposed to a problem with one device (busy, missing,
/// unsupported config, a stream-creation timeout).
fn is_host_level_kind(kind: ErrorKind) -> bool {
    matches!(
        kind,
        ErrorKind::StreamInvalidated | ErrorKind::HostUnavailable
    )
}

/// Whether `err` carries a host-level cpal error (see [`is_host_level_kind`]).
/// Only these justify dropping the cached PulseAudio connection: dropping a
/// healthy one leaks its client thread and socket, since the pulseaudio
/// client thread is not woken on drop.
pub fn is_host_level_error(err: &anyhow::Error) -> bool {
    err.chain()
        .filter_map(|e| e.downcast_ref::<cpal::Error>())
        .any(|e| is_host_level_kind(e.kind()))
}

/// Drop the cached PulseAudio connection so the next lookup reconnects.
pub fn invalidate_pulse_host() {
    pulse_slot().host = None;
}

/// Drop the cached connection and, for a capture, back off, after a fresh
/// one failed.
fn mark_pulse_failed(lookup: Lookup) {
    record_pulse_failure(&mut pulse_slot(), Instant::now(), lookup);
}

/// PulseAudio input devices. A cached connection that fails to list with a
/// host-level error (the server restarted) is replaced once with a fresh one;
/// any other error keeps the connection. The flag is true when the devices
/// came from the cached connection.
fn pulse_input_devices(lookup: Lookup) -> (Vec<(InputDeviceInfo, cpal::Device)>, bool) {
    let Some((host, cached)) = pulse_host(lookup) else {
        return (Vec::new(), false);
    };
    match input_devices(&host, DeviceHost::PulseAudio) {
        Ok(devices) => (devices, cached),
        Err(e) if is_host_level_kind(e.kind()) => {
            if !cached {
                debug!("failed to list PulseAudio input devices: {e}");
                mark_pulse_failed(lookup);
                return (Vec::new(), false);
            }
            debug!("cached PulseAudio connection failed ({e}), reconnecting");
            invalidate_pulse_host();
            let Some((host, _)) = pulse_host(lookup) else {
                return (Vec::new(), false);
            };
            match input_devices(&host, DeviceHost::PulseAudio) {
                Ok(devices) => (devices, false),
                Err(e) => {
                    debug!("failed to list PulseAudio input devices: {e}");
                    if is_host_level_kind(e.kind()) {
                        mark_pulse_failed(lookup);
                    }
                    (Vec::new(), false)
                }
            }
        }
        Err(e) => {
            debug!("failed to list PulseAudio input devices: {e}");
            (Vec::new(), false)
        }
    }
}

fn device_info(device: &cpal::Device, kind: DeviceHost) -> Option<InputDeviceInfo> {
    let id = device.id().ok()?.id().to_string();
    let description = device
        .description()
        .map(|d| d.name().to_string())
        .unwrap_or_else(|_| id.clone());
    Some(InputDeviceInfo {
        host: kind,
        id,
        description,
    })
}

fn input_devices(
    host: &cpal::Host,
    kind: DeviceHost,
) -> Result<Vec<(InputDeviceInfo, cpal::Device)>, cpal::Error> {
    Ok(host
        .input_devices()?
        .filter_map(|device| Some((device_info(&device, kind)?, device)))
        .collect())
}

fn alsa_input_devices(host: &cpal::Host) -> Vec<(InputDeviceInfo, cpal::Device)> {
    input_devices(host, DeviceHost::Alsa).unwrap_or_else(|e| {
        debug!("failed to list ALSA input devices: {e}");
        Vec::new()
    })
}

/// An ALSA PCM by name, accepting the numeric form (`plughw:1,0`) that
/// enumeration lists as `plughw:CARD=1,DEV=0`.
fn alsa_device_by_name(host: &cpal::Host, name: &str) -> Option<(InputDeviceInfo, cpal::Device)> {
    let device = host.device_by_id(&DeviceId::new(HostId::Alsa, name))?;
    if !device.supports_input() {
        return None;
    }
    Some((device_info(&device, DeviceHost::Alsa)?, device))
}

/// Input devices for the config editor and setup: PulseAudio sources when a
/// PulseAudio server is reachable, else ALSA PCMs. A diagnostic lookup: it
/// never starts the PulseAudio backoff.
pub fn list_input_devices() -> Vec<InputDeviceInfo> {
    let (devices, _) = pulse_input_devices(Lookup::Diagnostic);
    if !devices.is_empty() {
        return devices.into_iter().map(|(info, _)| info).collect();
    }
    list_alsa_input_devices()
}

/// Input devices for a daemon error message, given the configured
/// `[audio] device`. A stock `"default"` config never touches PulseAudio, so
/// an error path cannot open a connection the capture path would not have.
pub fn list_input_devices_for(configured: &str) -> Vec<InputDeviceInfo> {
    if is_default_name(configured) {
        list_alsa_input_devices()
    } else {
        list_input_devices()
    }
}

fn list_alsa_input_devices() -> Vec<InputDeviceInfo> {
    alsa_host()
        .map(|host| alsa_input_devices(&host))
        .unwrap_or_default()
        .into_iter()
        .map(|(info, _)| info)
        .collect()
}

/// Last unknown name we warned about, so a bad name warns once, not per
/// recording. Cleared when a named device resolves, so a later miss warns again.
static WARNED_UNKNOWN: Mutex<Option<String>> = Mutex::new(None);

/// Whether to warn that `name` matched no device. A capture lookup records
/// the name so the next recording stays quiet; a diagnostic one warns
/// without recording it.
fn should_warn_unknown(warned: &mut Option<String>, name: &str, lookup: Lookup) -> bool {
    if warned.as_deref() == Some(name) {
        return false;
    }
    if lookup == Lookup::Capture {
        *warned = Some(name.to_string());
    }
    true
}

/// Resolve `[audio] device` to an input device for a recording.
pub fn resolve_input(name: &str) -> anyhow::Result<ResolvedInput> {
    resolve(name, Lookup::Capture)
}

/// Resolve `[audio] device` for a diagnostic (the daemon's startup check).
/// Same result as [`resolve_input`], without its side effects: a failed
/// PulseAudio connection does not start the retry backoff, and an unknown
/// name is not marked as warned. The startup check can run before the sound
/// server is up, and the first recording must not inherit that.
pub fn resolve_input_for_diagnostics(name: &str) -> anyhow::Result<ResolvedInput> {
    resolve(name, Lookup::Diagnostic)
}

fn resolve(name: &str, lookup: Lookup) -> anyhow::Result<ResolvedInput> {
    if !is_default_name(name) {
        let (mut candidates, cached_pulse_host) = pulse_input_devices(lookup);
        let infos = |c: &[(InputDeviceInfo, cpal::Device)]| -> Vec<InputDeviceInfo> {
            c.iter().map(|(info, _)| info.clone()).collect()
        };
        let mut found =
            match_device(name, &infos(&candidates)).map(|index| candidates.swap_remove(index));
        if found.is_none() {
            if let Ok(host) = alsa_host() {
                candidates.extend(alsa_input_devices(&host));
                found = match_device(name, &infos(&candidates))
                    .map(|index| candidates.swap_remove(index))
                    .or_else(|| alsa_device_by_name(&host, name.trim()));
            }
        }
        if let Some((info, device)) = found {
            let (buffer_size, build_timeout) = match info.host {
                DeviceHost::PulseAudio => (
                    BufferSize::Fixed(PULSE_INPUT_FRAMES),
                    Some(PULSE_BUILD_TIMEOUT),
                ),
                DeviceHost::Alsa => (BufferSize::Fixed(ALSA_DEFAULT_INPUT_FRAMES), None),
            };
            *WARNED_UNKNOWN.lock().unwrap_or_else(|e| e.into_inner()) = None;
            return Ok(ResolvedInput {
                device,
                buffer_size,
                build_timeout,
                cached_pulse_host: cached_pulse_host && info.host == DeviceHost::PulseAudio,
                source: InputSource::Named(info.host, info.id),
            });
        }

        let mut warned = WARNED_UNKNOWN.lock().unwrap_or_else(|e| e.into_inner());
        if should_warn_unknown(&mut warned, name, lookup) {
            let pulse: Vec<&str> = candidates
                .iter()
                .filter(|(i, _)| i.host == DeviceHost::PulseAudio)
                .map(|(i, _)| i.id.as_str())
                .collect();
            let names = if pulse.is_empty() {
                candidates.iter().map(|(i, _)| i.id.as_str()).collect()
            } else {
                pulse
            };
            warn!(
                "[audio] device \"{name}\" not found, using the system default. \
                 Valid names: {} (or an ALSA PCM name from `arecord -L`)",
                names.join(", ")
            );
        }
    }

    let device = alsa_host()?
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("no default audio input device found"))?;
    Ok(ResolvedInput {
        device,
        buffer_size: BufferSize::Fixed(ALSA_DEFAULT_INPUT_FRAMES),
        build_timeout: None,
        cached_pulse_host: false,
        source: if is_default_name(name) {
            InputSource::Default
        } else {
            InputSource::Fallback
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(host: DeviceHost, id: &str, description: &str) -> InputDeviceInfo {
        InputDeviceInfo {
            host,
            id: id.to_string(),
            description: description.to_string(),
        }
    }

    fn candidates() -> Vec<InputDeviceInfo> {
        vec![
            dev(DeviceHost::Alsa, "pipewire", "PipeWire Sound Server"),
            dev(DeviceHost::Alsa, "sysdefault:CARD=sofhdadsp", "sof-hda-dsp"),
            dev(DeviceHost::Alsa, "shared", "Shared Mic"),
            dev(DeviceHost::PulseAudio, "alsa_input.usb-mic", "USB Mic"),
            dev(
                DeviceHost::PulseAudio,
                "alsa_output.speaker.monitor",
                "Monitor of Speaker",
            ),
            dev(DeviceHost::PulseAudio, "shared", "Pulse Shared"),
            dev(DeviceHost::PulseAudio, "other", "Shared Mic"),
        ]
    }

    #[test]
    fn default_names() {
        assert!(is_default_name("default"));
        assert!(is_default_name(""));
        assert!(is_default_name("   "));
        assert!(is_default_name(" default "));
        assert!(!is_default_name("pipewire"));
    }

    #[test]
    fn exact_pulse_id() {
        assert_eq!(match_device("alsa_input.usb-mic", &candidates()), Some(3));
        assert_eq!(
            match_device("  alsa_input.usb-mic ", &candidates()),
            Some(3)
        );
    }

    #[test]
    fn description_match() {
        assert_eq!(match_device("USB Mic", &candidates()), Some(3));
    }

    #[test]
    fn alsa_match() {
        assert_eq!(match_device("pipewire", &candidates()), Some(0));
        assert_eq!(
            match_device("sysdefault:CARD=sofhdadsp", &candidates()),
            Some(1)
        );
        assert_eq!(match_device("sof-hda-dsp", &candidates()), Some(1));
    }

    #[test]
    fn pulse_preferred_over_alsa() {
        assert_eq!(match_device("shared", &candidates()), Some(5));
        // Description match on PulseAudio still beats an ALSA description match.
        assert_eq!(match_device("Shared Mic", &candidates()), Some(6));
    }

    #[test]
    fn id_preferred_over_description() {
        let list = vec![
            dev(DeviceHost::PulseAudio, "a", "mic"),
            dev(DeviceHost::PulseAudio, "mic", "b"),
        ];
        assert_eq!(match_device("mic", &list), Some(1));
    }

    #[test]
    fn unknown_name() {
        assert_eq!(match_device("nope", &candidates()), None);
        assert_eq!(match_device("nope", &[]), None);
    }

    #[test]
    fn monitor_detection() {
        let list = candidates();
        assert!(list[4].is_monitor());
        assert!(!list[3].is_monitor());
        assert!(!dev(DeviceHost::Alsa, "x.monitor", "").is_monitor());
    }

    #[test]
    fn pulse_retry_backoff() {
        let now = Instant::now();
        assert!(pulse_retry_due(None, now));
        assert!(!pulse_retry_due(Some(now), now));
        assert!(!pulse_retry_due(
            Some(now),
            now + PULSE_RETRY_AFTER - Duration::from_millis(1)
        ));
        assert!(pulse_retry_due(Some(now), now + PULSE_RETRY_AFTER));
        // A clock that appears to go backwards does not unblock early.
        assert!(!pulse_retry_due(Some(now + Duration::from_secs(5)), now));
    }

    #[test]
    fn diagnostic_failure_does_not_back_off() {
        let now = Instant::now();
        let mut slot = PulseSlot {
            host: None,
            failed_at: None,
        };
        record_pulse_failure(&mut slot, now, Lookup::Diagnostic);
        assert_eq!(slot.failed_at, None);
        assert!(pulse_retry_due(slot.failed_at, now));

        record_pulse_failure(&mut slot, now, Lookup::Capture);
        assert_eq!(slot.failed_at, Some(now));
        assert!(!pulse_retry_due(slot.failed_at, now));

        // A diagnostic failure leaves an existing capture backoff alone.
        let later = now + Duration::from_secs(1);
        record_pulse_failure(&mut slot, later, Lookup::Diagnostic);
        assert_eq!(slot.failed_at, Some(now));
    }

    #[test]
    fn unknown_name_warns_once_per_capture() {
        let mut warned = None;
        assert!(should_warn_unknown(&mut warned, "bad", Lookup::Capture));
        assert!(!should_warn_unknown(&mut warned, "bad", Lookup::Capture));
        // A different name warns again.
        assert!(should_warn_unknown(&mut warned, "worse", Lookup::Capture));
        assert!(should_warn_unknown(&mut warned, "bad", Lookup::Capture));
    }

    #[test]
    fn diagnostic_warning_is_not_recorded() {
        let mut warned = None;
        assert!(should_warn_unknown(&mut warned, "bad", Lookup::Diagnostic));
        assert_eq!(warned, None);
        // The first recording after the startup check still warns.
        assert!(should_warn_unknown(&mut warned, "bad", Lookup::Capture));
        // A name already warned about by a capture stays quiet in diagnostics.
        assert!(!should_warn_unknown(&mut warned, "bad", Lookup::Diagnostic));
    }

    #[test]
    fn alsa_default_buffer_matches_cpal_015() {
        // cpal 0.18 opens 2 periods; cpal 0.15 asked for a 100 ms buffer.
        let ring_ms = 2 * ALSA_DEFAULT_INPUT_FRAMES * 1000 / crate::audio::capture::SAMPLE_RATE;
        assert_eq!(ring_ms, 100);
    }

    #[test]
    fn host_level_errors() {
        assert!(is_host_level_kind(ErrorKind::StreamInvalidated));
        assert!(is_host_level_kind(ErrorKind::HostUnavailable));
        // Device-specific: keep the cached connection.
        for kind in [
            ErrorKind::DeviceBusy,
            ErrorKind::DeviceNotAvailable,
            ErrorKind::UnsupportedConfig,
            ErrorKind::InvalidInput,
            ErrorKind::PermissionDenied,
        ] {
            assert!(!is_host_level_kind(kind), "{kind:?}");
        }

        // Found through anyhow context layers.
        let err = anyhow::Error::new(cpal::Error::with_message(
            ErrorKind::StreamInvalidated,
            "PulseAudio disconnected",
        ))
        .context("failed to build audio input stream");
        assert!(is_host_level_error(&err));
        let err = anyhow::Error::new(cpal::Error::with_message(
            ErrorKind::DeviceNotAvailable,
            "Stream creation timed out",
        ))
        .context("failed to build audio input stream");
        assert!(!is_host_level_error(&err));
        assert!(!is_host_level_error(&anyhow::anyhow!("plain")));
    }

    #[test]
    fn source_display() {
        assert_eq!(
            InputSource::Named(DeviceHost::PulseAudio, "x".into()).to_string(),
            "PulseAudio source x"
        );
        assert_eq!(
            InputSource::Fallback.to_string(),
            "system default (ALSA, fallback)"
        );
    }
}
