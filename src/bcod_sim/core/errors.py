"""Structured errors shared by resolution and conversion boundaries."""


class BCODSimError(Exception):
    """Base simulator error."""


class ConfigSchemaError(BCODSimError):
    pass


class UnknownReferenceError(BCODSimError):
    pass


class DuplicateIdentityError(BCODSimError):
    pass


class PhysicalValidationError(BCODSimError):
    pass


class FrameConversionError(BCODSimError):
    pass


class ExternalDataUnavailableError(BCODSimError):
    pass


class OperatingEnvelopeError(BCODSimError):
    pass


class NonFiniteStateError(BCODSimError):
    pass


class CommandBoundsError(BCODSimError):
    pass


class ExternalDataCoverageError(BCODSimError):
    pass


class DomainQueryError(BCODSimError):
    """A capability is missing or cannot be sampled at the requested point."""


class MissingCapabilityError(DomainQueryError):
    pass


class InvalidMediumError(DomainQueryError):
    pass


class CollisionSolverError(BCODSimError):
    pass


class PolicyContractMismatchError(BCODSimError):
    pass


class ObservationContractMismatchError(PolicyContractMismatchError):
    pass


class LoggingBackpressureError(BCODSimError):
    pass
