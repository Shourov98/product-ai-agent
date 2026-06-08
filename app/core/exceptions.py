from __future__ import annotations


class ProductNotFoundError(Exception):
    pass


class ProviderUnavailableError(Exception):
    pass


class PricingGuardrailError(Exception):
    pass
