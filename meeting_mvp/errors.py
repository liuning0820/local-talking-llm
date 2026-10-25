class MeetingError(Exception):
    """An actionable error raised by the meeting MVP."""

    def __init__(self, message: str, code: str = "MEETING_ERROR") -> None:
        super().__init__(message)
        self.code = code


class DeviceError(MeetingError):
    """An audio device cannot be used."""


class PrivacyError(MeetingError):
    """A network request is not permitted by local-only settings."""


class RecoverableJobError(MeetingError):
    """A job can be retried."""


class PermanentJobError(MeetingError):
    """A job cannot be retried automatically."""
