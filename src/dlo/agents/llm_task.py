"""
LLM Task
"""

import logging

from functools import cached_property

from json_schema_to_pydantic import create_model
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate

from dlo.agents.llm import ChatModelFactory
from dlo.common.exception import errors
from dlo.core.config import Profile, Project
from dlo.core.models.agent import AgentManifest, LLMTask

log = logging.getLogger("__name__")


class LLMTaskBuilder:
    def __init__(self, project: Project, profile: Profile, llm_task: LLMTask):
        self.project = project
        self.profile = profile

        self.llm_task = llm_task

    def get_model(self, model: str):
        model_provider, model = model.split("/")
        provider = self.profile.providers.get(model_provider)
        if provider is None:
            raise errors.DloParseError(f"Provider `{model_provider}` not found in the profile")

        config = {
            **provider.config,
            "provider": provider.provider,
            "model": model,
        }
        if self.llm_task.temperature is not None:
            config["temperature"] = self.llm_task.temperature
        if self.llm_task.reasoning_effort is not None:
            config["reasoning_effort"] = self.llm_task.reasoning_effort

        return ChatModelFactory.create(**config)

    @cached_property
    def chain(self):
        partial_variables = {}
        if self.llm_task.output_ is not None:
            # output_parser = JsonOutputParser()

            OutputModel = create_model(self.llm_task.output_)
            output_parser = PydanticOutputParser(pydantic_object=OutputModel)
            partial_variables["output_format"] = output_parser.get_format_instructions()
        else:
            output_parser = StrOutputParser()

        prompt = PromptTemplate(
            template=self.llm_task.prompt,
            input_variables=self.llm_task.input_,
            partial_variables=partial_variables,
        )

        llm = self.get_model(self.llm_task.primary_model)

        return prompt | llm | output_parser

    async def ainvoke(self, *args, **kwargs):
        return await self.chain.ainvoke(*args, **kwargs)


class LLMTaskRegistry:
    def __init__(self, project: Project, profile: Profile, agent_manifest: AgentManifest):
        self.project = project
        self.profile = profile

        self.agent_manifest = agent_manifest

    def llm_task(self, name: str):
        llm_task = self.agent_manifest.llm_tasks.get(name)
        if llm_task is None:
            raise errors.MethodNotFoundError(f"unknown tools type {name!r}")
        return LLMTaskBuilder(
            project=self.project,
            profile=self.profile,
            llm_task=llm_task,
        )
