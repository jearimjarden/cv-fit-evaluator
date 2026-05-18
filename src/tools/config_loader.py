import yaml
from pathlib import Path
import os
from pydantic import ValidationError
from .schemas import Config, Env, AuthConfig
from .exceptions_schemas import InvalidConfig, InvalidSettings


def load_config():
    try:
        path = Path("config.yaml")

        with open(path, "r") as file:
            config_data = yaml.safe_load(file)

        return Config(**config_data)

    except FileNotFoundError as e:
        raise InvalidConfig(f"Configuration was not found at path: {path}") from e

    except ValidationError as e:
        messages = []

        for err in e.errors():
            field = ".".join(str(x) for x in err["loc"])

            if err["type"] == "missing":
                messages.append(f"Missing config parameter: '{field}'")

            elif err["type"] == "extra_forbidden":
                messages.append(f"Forbidden extra config parameter: '{field}'")

            else:
                messages.append(f"Invalid config value for '{field}': {err['msg']}")

        raise InvalidConfig(" | ".join(messages)) from e


def load_env():
    try:
        env = Env()  # type: ignore
        os.environ["HF_TOKEN"] = env.hf_api_key
        return env

    except ValidationError as e:
        messages = []

        for err in e.errors():
            field = ".".join(str(x) for x in err["loc"])

            if err["type"] == "missing":
                messages.append(f"Missing env parameter: '{field}'")

            elif err["type"] == "extra_forbidden":
                messages.append(f"Forbidden extra env parameter: '{field}'")

            else:
                messages.append(f"Invalid env value for '{field}': {err['msg']}")

        raise InvalidSettings(" | ".join(messages)) from e


def load_auth_config():
    try:
        path = Path("auth_config.yaml")

        with open(path, "r") as file:
            config_data = yaml.safe_load(file)

        return AuthConfig(**config_data)

    except FileNotFoundError as e:
        raise InvalidConfig(
            f"Authentication config was not found at path: {path}"
        ) from e

    except ValidationError as e:
        messages = []

        for err in e.errors():
            field = ".".join(str(x) for x in err["loc"])

            if err["type"] == "missing":
                messages.append(f"Missing auth_config parameter: '{field}'")

            elif err["type"] == "extra_forbidden":
                messages.append(f"Forbidden auth_config parameter: '{field}'")

            else:
                messages.append(
                    f"Invalid auth_config value for '{field}': {err['msg']}"
                )

        raise InvalidConfig(" | ".join(messages)) from e
