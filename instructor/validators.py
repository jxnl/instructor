from typing import Callable, Any, TypeVar, Optional
from inspect import signature
from pydantic import ValidationInfo, BaseModel
from collections.abc import Awaitable
import asyncio

ASYNC_VALIDATOR_KEY = "__async_validator__"
ASYNC_MODEL_VALIDATOR_KEY = "__async_model_validator__"
T = TypeVar("T", bound=Callable[..., Any])


class AsyncValidationContext:
    context: dict[str, Any]

    def __init__(self, context: dict[str, Any]):
        self.context = context


class AsyncValidationError(ValueError):
    errors: list[ValueError]


def async_field_validator(field: str, *fields: str) -> Callable[[T], T]:
    field_names = field, *fields

    def decorator(func: T) -> T:
        params = signature(func).parameters
        requires_validation_context = False
        if len(params) == 3:
            if "info" not in params:
                raise ValueError(
                    "Async validator can only have a value parameter and an optional info parameter"
                )
            if params["info"].annotation != ValidationInfo:
                raise ValueError(
                    "Async validator info parameter must be of type ValidationInfo"
                )
            requires_validation_context = True

        setattr(
            func, ASYNC_VALIDATOR_KEY, (field_names, func, requires_validation_context)
        )
        return func

    return decorator


def async_model_validator() -> Callable[[T], T]:
    def decorator(func: T) -> T:
        params = signature(func).parameters
        requires_validation_context = False
        if len(params) > 2:
            raise ValueError("Invalid Parameter Count!")

        if len(params) == 2:
            if "info" not in params:
                raise ValueError(
                    "Async validator can only have a value parameter and an optional info parameter"
                )
            if params["info"].annotation != ValidationInfo:
                raise ValueError(
                    "Async validator info parameter must be of type ValidationInfo"
                )
            requires_validation_context = True

        setattr(
            func,
            ASYNC_MODEL_VALIDATOR_KEY,
            (func, requires_validation_context),
        )
        return func

    return decorator


class AsyncInstructMixin:
    @classmethod
    def get_async_validators(cls):
        validators = [
            getattr(cls, name)
            for name in dir(cls)
            if hasattr(getattr(cls, name), ASYNC_VALIDATOR_KEY)
        ]
        return validators

    @classmethod
    def get_async_model_validators(cls):
        validators = [
            getattr(cls, name)
            for name in dir(cls)
            if hasattr(getattr(cls, name), ASYNC_MODEL_VALIDATOR_KEY)
        ]
        return validators

    def has_async_validators(self) -> bool:
        has_validators = (
            len(self.__class__.get_async_validators()) > 0
            or len(self.get_async_model_validators()) > 0
        )

        for _, attribute_value in self.__dict__.items():
            if isinstance(attribute_value, AsyncInstructMixin):
                has_validators = (
                    has_validators or attribute_value.has_async_validators()
                )

                # List of items too
            if isinstance(attribute_value, (list, set, tuple)):
                for item in attribute_value:  # type:ignore
                    if isinstance(item, AsyncInstructMixin):
                        has_validators = has_validators or item.has_async_validators()

        return has_validators

    async def execute_field_validator(
        self,
        func: Any,
        value: Any,
        context: Optional[AsyncValidationContext] = None,
        prefix: list[str] = [],
    ):
        try:
            if not context:
                await func(self, value)
            else:
                await func(self, value, context)

        except Exception as e:
            prefix_path = f" at {'.'.join(prefix)}" if prefix else ""
            return ValueError(f"Exception of {e} encountered{prefix_path}")

    async def execute_model_validator(
        self,
        func: Any,
        context: Optional[AsyncValidationContext] = None,
        prefix: list[str] = [],
    ):
        try:
            if not context:
                await func(self)
            else:
                await func(self, context)

        except Exception as e:
            prefix_path = f" at {'.'.join(prefix)}" if prefix else ""
            return ValueError(f"Exception of {e} encountered{prefix_path}")

    async def get_model_coroutines(
        self, validation_context: dict[str, Any] = {}, prefix: list[str] = []
    ) -> list[Awaitable[Optional[ValueError]]]:
        assert isinstance(
            self, BaseModel
        ), "AsyncInstructMixin must be applied to a BaseModel"
        values = dict(self)
        validators = self.__class__.get_async_validators()
        model_validators = self.get_async_model_validators()
        coros: list[Awaitable[Any]] = []
        for validator in validators + model_validators:
            # Model Validator
            if not hasattr(validator, ASYNC_VALIDATOR_KEY):
                validation_func, requires_validation_context = getattr(
                    validator, ASYNC_MODEL_VALIDATOR_KEY
                )
                coros.append(
                    self.execute_model_validator(
                        validation_func,
                        AsyncValidationContext(context=validation_context)
                        if requires_validation_context
                        else None,
                        prefix,
                    )
                )
            else:
                fields, validation_func, requires_validation_context = getattr(
                    validator, ASYNC_VALIDATOR_KEY
                )
                for field in fields:
                    if field not in values:
                        raise ValueError(f"Invalid Field of {field} provided")

                    coros.append(
                        self.execute_field_validator(
                            validation_func,
                            values[field],
                            AsyncValidationContext(context=validation_context)
                            if requires_validation_context
                            else None,
                            prefix=prefix + [field],
                        )
                    )

        for attribute_name, attribute_value in self.__dict__.items():
            # Supporting Sub Array
            if isinstance(attribute_value, AsyncInstructMixin):
                coros.extend(
                    await attribute_value.get_model_coroutines(
                        validation_context, prefix=prefix + [attribute_name]
                    )
                )

            # List of items too
            if isinstance(attribute_value, (list, set, tuple)):
                for item in attribute_value:  # type:ignore
                    if isinstance(item, AsyncInstructMixin):
                        coros.extend(
                            await item.get_model_coroutines(
                                validation_context, prefix=prefix + [attribute_name]
                            )
                        )

        return coros

    async def model_async_validate(
        self, validation_context: dict[str, Any] = {}
    ) -> list[ValueError]:
        coros = await self.get_model_coroutines(validation_context)
        return [item for item in await asyncio.gather(*coros) if item]
