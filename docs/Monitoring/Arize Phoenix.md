# Enabling LM Tracing with Arize Phoenix

Monitoring is important for several reasons, especially in the context of machine learning and software development. It helps identify issues and bugs in your Synalinks programs by providing a detailed log of events and operations. This makes it easier to pinpoint where things went wrong and why.

In this guide we are going to setup the tracing locally, for more information on how to setup in the cloud, refer to [Arize Phoenix documentation](https://docs.arize.com/phoenix)

```shell
uv pip install openinference-instrumentation-litellm arize-otel
```

To activate the LM tracing, add the following lines to the top of your script

```python
# Import open-telemetry dependencies
from arize.otel import register
from openinference.instrumentation.litellm import LiteLLMInstrumentor

# Setup OTel via Arize Phoenix convenience function
tracer_provider = register(
    space_id = "your-space-id", # in app space settings page
    api_key = "your-api-key", # in app space settings page
    project_name = "your-project-name", # name this to whatever you would like
)

LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
```

You are done, Arize Phoenix is now configured.

To launch Arize Phoenix server, first pull the docker image with the following command.

```shell
docker pull arizephoenix/phoenix
```

Then use the following command in a shell

```shell
docker run -p 6006:6006 -p 4317:4317 -i -t arizephoenix/phoenix:latest
```

Finally go to `http://0.0.0.0:6006` to monitor your application.