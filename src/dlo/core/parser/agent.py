import logging

from functools import cached_property
from pathlib import Path

from frontmatter import Post

from dlo.common.exception import errors
from dlo.core.config import Project
from dlo.core.models.agent import (
    AGENT_RESOURCE_REGISTRY,
    AgentManifest,
    AgentResource,
    Skills,
)
from dlo.core.parser.file_reader import FileReaderFromFileSystem

# Configure module logger
log = logging.getLogger(__name__)


class AgentManifestLoader:
    """Loads agent resources (agents, tools, tasks) from project files.

    Scans markdown files in agent directories and parses them into
    agent manifest resources based on frontmatter metadata.
    """

    def __init__(self, project: Project):
        self.project = project
        self.agent_manifest = AgentManifest.__from_project__(project)

    def resource_factory(self, file_path: str, md_data: Post, default_model, default_key: str):
        """Create agent resource from markdown file data.

        Args:
            file_path: Path to the markdown file.
            md_data: Parsed frontmatter Post object.
            default_model: Default model class for this resource type.
            default_key: Default manifest key for storing resource.

        Returns:
            Tuple of (resource instance, manifest key).

        Raises:
            DloCompilationError: If resource parsing fails.
        """
        model = default_model
        key = default_key

        try:
            resource_type = md_data.metadata.get("resource_type")
            resource = AgentResource.get_resource(resource_type)
            if resource:
                model = resource["model"]
                key = resource["key"]

            resource = model.from_dict({
                "name": file_path.stem,
                **md_data.metadata,
                "file_path": file_path,
                "prompt": md_data.content,
                "base_dir": self.agent_manifest.base_dir,
            })
        except Exception as e:
            available = ", ".join(AgentResource.available_types())
            raise errors.DloCompilationError(
                f"Error while parsing file {file_path.absolute().as_posix()}: {e}\n"
                f"Available resource types: {available}"
            ) from e

        return resource, key

    def load_for_dir(self, directory: str, default_model, default_key: str) -> None:
        """Load and parse all markdown files in a directory.

        Args:
            directory: Directory path to scan.
            default_model: Model class for resources in this directory.
            default_key: Manifest key for storing resources.
        """
        # Initalize file reader for the agent root
        reader = FileReaderFromFileSystem(directory)

        log.info("Found %d files in directory: %s", len(reader.files), directory)
        # Iterate over all files and parse them based on their type
        parsed_count = 0

        for file in reader.files:
            file_path = Path(file)

            if file_path.suffix != ".md":
                continue

            try:
                data = reader.read_markdown(file_path)

                resource, key = self.resource_factory(
                    file_path=file_path,
                    default_model=default_model,
                    md_data=data,
                    default_key=default_key,
                )
            except Exception as e:
                raise errors.DloCompilationError(
                    f"Error while parsing file {file_path.absolute().as_posix()}: {e}"
                ) from e

            agent_manifest_resource = getattr(self.agent_manifest, key)
            agent_manifest_resource[resource.unique_id] = resource

            parsed_count += 1

        log.info("Finished parsing agent files. Parsed: %d", parsed_count)

    @cached_property
    def skills_dir(self) -> Path:
        return self.agent_manifest.root_dir / "skills"

    def load_skills(self):
        if not self.skills_dir.is_dir():
            return
        skills = {}
        for sk in self.skills_dir.iterdir():
            if sk.is_dir():
                sk_name = sk.name
                skill = Skills(name=sk_name, path=f"skills/{sk_name}")
                skills[skill.unique_id] = skill

        self.agent_manifest.skills = skills

    def load(self) -> AgentManifest:
        """Load all agent resources defined in AGENT_RESOURCE_REGISTRY."""
        log.info("Starting Agent load for project: %s", self.project.project_root)

        # Load agents, tools, llm_tasks
        for config in AGENT_RESOURCE_REGISTRY:
            directory = self.agent_manifest.root_dir / config.directory_name

            if not directory.exists():
                log.debug(
                    "Skipping %s - directory not found: %s", config.resource_type.value, directory
                )
                continue

            log.info("Loading %s from %s", config.resource_type.value, directory)
            self.load_for_dir(
                directory=directory,
                default_model=config.model_class,
                default_key=config.manifest_key,
            )

        # Load skills
        self.load_skills()

        log.debug("Final agents state: %s", self.agent_manifest)

        return self.agent_manifest
