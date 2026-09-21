"""Strict policy bundle loading and hash validation."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from bcod_sim.config.hashing import content_hash
from bcod_sim.core.errors import PolicyContractMismatchError


REQUIRED_FILES = ("manifest.json", "model.onnx", "observation_contract.json", "action_contract.json",
                  "preprocessing.json", "normalization.json", "training_provenance.json")


class PolicyManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    policy_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    model_format: str
    model_sha256: str
    observation_contract_hash: str
    action_contract_hash: str
    simulator_commit: str = Field(min_length=1)
    training_config_hash: str = Field(min_length=1)
    trainer_commit: str = Field(min_length=1)
    trainer_version: str = Field(min_length=1)
    seeds: tuple[int, ...] = Field(min_length=1)
    preprocessing_sha256: str
    normalization_provenance: str = Field(min_length=1)
    training_provenance_sha256: str

    @field_validator("model_sha256", "observation_contract_hash", "action_contract_hash",
                     "training_config_hash", "preprocessing_sha256", "normalization_provenance",
                     "training_provenance_sha256")
    @classmethod
    def hashes(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("Expected lowercase SHA-256")
        return value


@dataclass(frozen=True)
class PolicyBundle:
    root: Path
    manifest: PolicyManifest
    observation_contract: dict
    action_contract: dict
    preprocessing: dict
    normalization: dict
    training_provenance: dict

    @classmethod
    def load(cls, root: str | Path, *, expected_observation_hash: str,
             expected_action_hash: str) -> "PolicyBundle":
        root = Path(root)
        missing = [name for name in REQUIRED_FILES if not (root/name).is_file()]
        if missing:
            raise PolicyContractMismatchError(f"Policy bundle missing required files: {missing}")
        try:
            manifest = PolicyManifest.model_validate_json((root/"manifest.json").read_text())
            documents = [json.loads((root/name).read_text()) for name in REQUIRED_FILES[2:]]
        except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
            raise PolicyContractMismatchError("Malformed policy bundle") from exc
        if manifest.model_format != "onnx":
            raise PolicyContractMismatchError("Public policy runtime requires ONNX model format")
        if any(not isinstance(document, dict) for document in documents):
            raise PolicyContractMismatchError("Policy metadata files must contain JSON objects")
        model_hash = hashlib.sha256((root/"model.onnx").read_bytes()).hexdigest()
        observation_hash, action_hash = content_hash(documents[0]), content_hash(documents[1])
        if (not isinstance(documents[0].get("model_inputs"), list) or
                not isinstance(documents[1].get("model_outputs"), list) or
                not documents[0]["model_inputs"] or not documents[1]["model_outputs"] or
                len(documents[0]["model_inputs"]) != len(set(documents[0]["model_inputs"])) or
                len(documents[1]["model_outputs"]) != len(set(documents[1]["model_outputs"]))):
            raise PolicyContractMismatchError("Policy contracts require unique model input/output bindings")
        if model_hash != manifest.model_sha256:
            raise PolicyContractMismatchError("Policy model checksum mismatch")
        if observation_hash != manifest.observation_contract_hash or observation_hash != expected_observation_hash:
            raise PolicyContractMismatchError("Observation contract hash mismatch")
        if action_hash != manifest.action_contract_hash or action_hash != expected_action_hash:
            raise PolicyContractMismatchError("Action contract hash mismatch")
        if content_hash(documents[3]) != manifest.normalization_provenance:
            raise PolicyContractMismatchError("Normalization provenance hash mismatch")
        if content_hash(documents[2]) != manifest.preprocessing_sha256:
            raise PolicyContractMismatchError("Preprocessing checksum mismatch")
        if content_hash(documents[4]) != manifest.training_provenance_sha256:
            raise PolicyContractMismatchError("Training provenance checksum mismatch")
        if any(seed < 0 for seed in manifest.seeds):
            raise PolicyContractMismatchError("Policy training seeds must be nonnegative")
        expected_training = {"simulator_commit": manifest.simulator_commit,
            "training_config_hash": manifest.training_config_hash, "trainer_commit": manifest.trainer_commit,
            "trainer_version": manifest.trainer_version, "seeds": list(manifest.seeds)}
        if any(documents[4].get(key) != value for key, value in expected_training.items()):
            raise PolicyContractMismatchError("Training provenance disagrees with policy manifest")
        return cls(root, manifest, *documents)
