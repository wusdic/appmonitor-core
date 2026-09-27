"""Accessors over the model.class layout (B02 owns the writer)."""
from helpers import make_store

from app.engines.behavior.lib import m_class
from app.engines.behavior.lib.classkeys import ORG


def _model():
    assign = {f"erp|10.0.0.{i}": {"role": "r1", "sub": "s1", "prob": 0.9, "static": ["office"],
                                  "pool": None} for i in range(1, 5)}
    assign["erp|10.0.0.9"] = {"role": "r2", "prob": 0.8, "static": [], "pool": "10.0.9.0/24"}
    assign["erp|10.0.0.8"] = {"role": "unique", "prob": 1.0}
    assign["oa|10.0.0.1"] = {"role": "r1", "prob": 0.95}
    return {"assign": assign, "roles": {"r1": {"name": "human/orders"}}, "version": 3}


def test_backoff_key_and_members():
    st = make_store()
    st.put_model(ORG[0], ORG[1], "model.class", _model())
    assert m_class.class_key(st, "erp", "10.0.0.2") == "class:r1"
    assert m_class.class_key(st, "erp", "10.0.0.9") is None          # role too small in erp
    assert m_class.class_key(st, "erp", "10.0.0.8") is None          # unique
    assert m_class.class_key(st, "oa", "10.0.0.1") is None           # 1 member in oa
    assert m_class.members(st, "erp", "r1") == [f"10.0.0.{i}" for i in range(1, 5)]
    assert m_class.role_name(st, "r1") == "human/orders" and m_class.version(st) == 3


def test_all_class_keys_and_members():
    st = make_store()
    st.put_model(ORG[0], ORG[1], "model.class", _model())
    keys = m_class.all_class_keys(st, "erp")
    assert "class:r1" in keys and "class:static:office" in keys and "class:pool:10.0.9.0/24" in keys
    assert "class:r2" not in keys
    assert m_class.class_members(st, "erp", "class:static:office") == [f"10.0.0.{i}" for i in range(1, 5)]
    assert m_class.class_members(st, "erp", "class:pool:10.0.9.0/24") == ["10.0.0.9"]


def test_empty_store():
    st = make_store()
    assert m_class.class_key(st, "erp", "10.0.0.1") is None and m_class.all_class_keys(st, "erp") == []
