import copy
import json
from dataclasses import dataclass, fields, is_dataclass


def dict_to_str(some_dict):
    return json.dumps(some_dict, sort_keys=True)


def str_to_dict(some_str):
    return json.loads(some_str)


@dataclass
class ExpConf:
    """Base class for serialization needed to save configuration objects.

    Add to a dataclass to benefit from the automated definitions of __init__,
    __str__ and __repr__. Will serialize based on configuration parameters
    passed to __init__ and ignore __post_init__ fields defined with

        param = field(init=False)
    """

    def as_str(self):
        return dict_to_str(self.as_dict())

    def as_dict(self):
        """Gets a dictionary of the init-relevant attributes.

        Can be used to copy a child of Serializable using dataclasses by
        calling ``Class(**obj.as_dict())``.

        Returns:
            A dictionary representation of the dataclass
        """

        def _serialize(obj):
            """Gets a representation of obj.

            Essentially a re-implementation of DataClass's ``asdict``
            except ignoring non-init fields.

            Args:
                obj: The object to serialize

            Returns:
                A simple structure representing the object
            """
            if isinstance(obj, ExpConf) and is_dataclass(obj):
                return {
                    f.name: _serialize(getattr(obj, f.name))
                    for f in fields(obj)
                    if f.init
                }
            elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
                return type(obj)(*[_serialize(v) for v in obj])
            elif isinstance(obj, (list, tuple)):
                return type(obj)(_serialize(v) for v in obj)
            elif isinstance(obj, dict):
                return type(obj)((_serialize(k), _serialize(v)) for k, v in obj.items())
            else:
                return copy.deepcopy(obj)

        return _serialize(self)

    @classmethod
    def from_dict(cls, some_dict):
        if not is_dataclass(cls) or not issubclass(cls, ExpConf):
            raise TypeError(
                "from_dict() should be called on Serializable dataclass instances"
            )
        return cls(
            **{
                f.name: (
                    f.type(**some_dict[f.name])
                    if issubclass(f.type, ExpConf)
                    else some_dict[f.name]
                )
                for f in fields(cls)
                if f.init
            }
        )

    @classmethod
    def from_str(cls, some_str):
        return cls.from_dict(str_to_dict(some_str))
