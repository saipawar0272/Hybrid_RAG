import pytest

from src import config

def test_valid_configuration(monkeypatch):
    """Valid configuration should not raise an exception."""

    monkeypatch.setattr(config, "CHUNK_SIZE", 700)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 120)
    monkeypatch.setattr(config, "DENSE_TOP_K", 8)
    monkeypatch.setattr(config, "SPARSE_TOP_K",8)
    monkeypatch.setattr(config, "FINAL_TOP_K", 5)
    monkeypatch.setattr(config, "RRF_C", 60)
    monkeypatch.setattr(config, "RRF_DENSE_WEIGHT", 0.5)
    monkeypatch.setattr(config, "RRF_SPARSE_WEIGHT", 0.5)
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "test-embedding")
    monkeypatch.setattr(config, "GROQ_MODEL", "test-llm")

    config.validate_config()


def test_chunk_size_must_be_positive(monkeypatch):
    """Zero chunk size shoul raise ValueError."""

    monkeypatch.setattr(config, "CHUNK_SIZE", 0)

    with pytest.raises(ValueError):
        config.validate_config()


def test_chunk_overlap_must_be_less_than_chunk_size(monkeypatch):
    """Overlap equal to chunk size should raise ValueError."""

    monkeypatch.setattr(config, "CHUNK_SIZE", 700)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 700)

    with pytest.raises(ValueError):
        config.validate_config()


def test_rrf_weights_must_sum_to_positive_value(monkeypatch):
    """Both RRF weight set to zero should raise ValueError."""

    monkeypatch.setattr(config, "RRF_DENSE_WEIGHT", 0.0)
    monkeypatch.setattr(config, "RRF_SPARSE_WEIGHT", 0.0)

    with pytest.raises(ValueError):
        config.validate_config()




def test_negative_dense_weight_is_invalid(monkeypatch):
    monkeypatch.setattr(config, "RRF_DENSE_WEIGHT", -0.1)

    with pytest.raises(ValueError):
        config.validate_config()


def test_rrf_weight_greater_than_one_is_invalid(monkeypatch):
    monkeypatch.setattr(config, "RRF_SPARSE_WEIGHT", 1.2)

    with pytest.raises(ValueError):
        config.validate_config()

