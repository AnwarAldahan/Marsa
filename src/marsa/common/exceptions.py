"""Project-specific exceptions."""


class MarsaError(Exception):
    """Base exception for Marsa."""


class ConfigError(MarsaError):
    """Configuration is missing or invalid."""


class DataNotReadyError(MarsaError):
    """A required data artifact has not been generated yet."""


class ModelNotReadyError(MarsaError):
    """A required model artifact has not been trained or registered yet."""


class SyntheticDataNotReadyError(DataNotReadyError):
    """Synthetic operational data is required but has not been generated."""


class NotReadyError(MarsaError):
    """A scaffolded phase is intentionally not implemented yet."""

