import os
from pathlib import Path

from premsql.agents import BaseLineAgent
from premsql.agents.tools import SimpleMatplotlibTool
from premsql.executors import ExecutorUsingLangChain
from premsql.generators import Text2SQLGeneratorOpenAI
from premsql.playground import AgentServer


def load_env_file_if_present() -> None:
    # Works in Codespaces and local runs without requiring `export` for each var.
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    load_env_file_if_present()

    openai_api_key = get_required_env("OPENAI_API_KEY")
    db_path = get_required_env("PREMSQL_DB_PATH")

    session_name = os.getenv("PREMSQL_SESSION_NAME", "kaggle_test_session")
    model_name = os.getenv("PREMSQL_MODEL_NAME", "gpt-5-mini")
    port = int(os.getenv("PREMSQL_AGENT_PORT", "8100"))
    host = os.getenv("PREMSQL_AGENT_HOST", "0.0.0.0")

    db_connection_uri = f"sqlite:///{db_path}"

    text2sql_model = Text2SQLGeneratorOpenAI(
        model_name=model_name,
        experiment_name="openai_text2sql_model",
        type="test",
        openai_api_key=openai_api_key,
    )

    analyser_plotter_model = Text2SQLGeneratorOpenAI(
        model_name=model_name,
        experiment_name="openai_analysis_model",
        type="test",
        openai_api_key=openai_api_key,
    )

    agent = BaseLineAgent(
        session_name=session_name,
        db_connection_uri=db_connection_uri,
        specialized_model1=text2sql_model,
        specialized_model2=analyser_plotter_model,
        executor=ExecutorUsingLangChain(),
        auto_filter_tables=False,
        plot_tool=SimpleMatplotlibTool(),
    )

    agent_server = AgentServer(agent=agent, url=host, port=port)
    agent_server.launch()


if __name__ == "__main__":
    main()
