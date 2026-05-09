"""Tests for pipeGEM/utils/_class.py — ObjectFactory, is_iter, classproperty."""
import pytest

from pipeGEM.utils._class import ObjectFactory, is_iter, classproperty


# =====================================================================
# ObjectFactory
# =====================================================================

class TestObjectFactory:

    def test_register_and_create(self):
        factory = ObjectFactory()
        factory.register("int", int)
        result = factory.create("int")
        assert result == 0  # int() == 0

    def test_unregistered_raises_keyerror(self):
        factory = ObjectFactory()
        with pytest.raises(KeyError):
            factory.create("missing")

    def test_items_returns_registered(self):
        factory = ObjectFactory()
        factory.register("a", int)
        factory.register("b", str)
        items = dict(factory.items())
        assert "a" in items
        assert "b" in items
        assert items["a"] is int

    def test_getitem_works(self):
        factory = ObjectFactory()
        factory.register("x", list)
        assert factory["x"] is list

    def test_getitem_missing_raises(self):
        factory = ObjectFactory()
        with pytest.raises(KeyError):
            _ = factory["missing"]

    def test_create_with_kwargs(self):
        factory = ObjectFactory()
        factory.register("dict", dict)
        result = factory.create("dict", a=1, b=2)
        assert result == {"a": 1, "b": 2}


# =====================================================================
# is_iter
# =====================================================================

class TestIsIter:

    def test_list_is_iterable(self):
        assert is_iter([1, 2, 3]) is True

    def test_int_is_not_iterable(self):
        assert is_iter(42) is False

    def test_string_is_iterable(self):
        assert is_iter("hello") is True

    def test_generator_is_iterable(self):
        gen = (x for x in range(3))
        assert is_iter(gen) is True

    def test_none_is_not_iterable(self):
        assert is_iter(None) is False

    def test_dict_is_iterable(self):
        assert is_iter({"a": 1}) is True


# =====================================================================
# classproperty
# =====================================================================

class TestClassProperty:

    def test_accessible_on_class(self):
        class MyClass:
            _value = 42

            @classproperty
            def value(cls):
                return cls._value

        assert MyClass.value == 42

    def test_accessible_on_instance(self):
        class MyClass:
            _value = 99

            @classproperty
            def value(cls):
                return cls._value

        obj = MyClass()
        assert obj.value == 99
