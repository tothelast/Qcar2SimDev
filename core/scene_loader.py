#!/usr/bin/env python3
"""
Scene Loader Module

Loads scene definitions and actor definitions from JSON files.
Scenes reference actors by name, and actors are loaded from the actor library.

Directory structure:
  config/
    routes/          - Route definitions
    actors/          - Actor library
      vehicles/      - Autonomous vehicles
      pedestrians/   - Pedestrians
      static/        - Parked vehicles, stop signs
    scenes/          - Scene compositions
      training/      - Training scenes
      testing/       - Testing scenes
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any


class ActorDefinition:
    """Represents a single actor loaded from the actor library."""

    def __init__(self, actor_data: Dict[str, Any], actor_path: str):
        """
        Initialize actor definition from loaded JSON data.

        Args:
            actor_data: Dictionary containing actor configuration
            actor_path: Path to the actor file (for error reporting)
        """
        self.actor_path = actor_path
        self.name = actor_data.get('name', 'unnamed_actor')
        self.description = actor_data.get('description', '')
        self.type = actor_data.get('type', 'unknown')
        self.actor_number = actor_data.get('actor_number', 0)

        # Store all actor data for easy access
        self.data = actor_data

    def __repr__(self):
        return f"ActorDefinition(name='{self.name}', type='{self.type}', actor_number={self.actor_number})"


class SceneDefinition:
    """
    Represents a complete scene definition loaded from JSON.

    A scene includes:
    - Ego vehicle route (by name)
    - List of actor names to spawn
    - Actors are loaded from the actor library
    """

    def __init__(self, scene_data: Dict[str, Any], scene_path: str, actors: List[ActorDefinition]):
        """
        Initialize scene definition from loaded JSON data.

        Args:
            scene_data: Dictionary containing scene configuration
            scene_path: Path to the scene file (for error reporting)
            actors: List of loaded ActorDefinition objects
        """
        self.scene_path = scene_path
        self.name = scene_data.get('name', 'unnamed_scene')
        self.description = scene_data.get('description', '')
        self.ego_route = scene_data.get('ego_route', '')

        # Store loaded actors
        self.actors = actors

        # Categorize actors by type for easy access
        self.autonomous_vehicles = [a for a in actors if a.type == 'autonomous_vehicle']
        self.pedestrians = [a for a in actors if a.type == 'pedestrian']
        self.parked_vehicles = [a for a in actors if a.type == 'parked_vehicle']
        self.stop_signs = [a for a in actors if a.type == 'stop_sign']

    def validate(self) -> tuple[bool, str]:
        """
        Validate scene definition.

        Returns:
            (is_valid, error_message) tuple
        """
        if not self.ego_route:
            return False, f"Scene '{self.name}' missing ego_route"

        # All actors are pre-validated when loaded from actor library
        return True, ""

    def __str__(self) -> str:
        """String representation of scene."""
        actors_summary = []
        if self.autonomous_vehicles:
            actors_summary.append(f"{len(self.autonomous_vehicles)} autonomous vehicle(s)")
        if self.pedestrians:
            actors_summary.append(f"{len(self.pedestrians)} pedestrian(s)")
        if self.parked_vehicles:
            actors_summary.append(f"{len(self.parked_vehicles)} parked vehicle(s)")
        if self.stop_signs:
            actors_summary.append(f"{len(self.stop_signs)} stop sign(s)")

        actors_str = ", ".join(actors_summary) if actors_summary else "no actors"
        return f"Scene '{self.name}': route={self.ego_route}, {actors_str}"


class SceneLoader:
    """
    Loads and manages scene definitions and actor library from JSON files.

    Directory structure:
      config/
        routes/          - Route definitions
        actors/          - Actor library
        scenes/          - Scene compositions
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize scene loader.

        Args:
            config_dir: Path to config directory (default: ./config)
        """
        if config_dir is None:
            # Default to config/ directory relative to project root
            project_root = Path(__file__).parent.parent
            config_dir = project_root / 'config'

        self.config_dir = Path(config_dir)
        self.actors_dir = self.config_dir / 'actors'
        self.scenes_dir = self.config_dir / 'scenes'
        self.routes_dir = self.config_dir / 'routes'

        # Cache for loaded actors
        self._actor_cache: Dict[str, ActorDefinition] = {}

    def load_actor(self, actor_name: str) -> Optional[ActorDefinition]:
        """
        Load an actor by name from the actor library.

        Args:
            actor_name: Name of the actor (without .json extension)

        Returns:
            ActorDefinition object or None if not found
        """
        # Check cache first
        if actor_name in self._actor_cache:
            return self._actor_cache[actor_name]

        # Search in all actor subdirectories
        for subdir in ['vehicles', 'pedestrians', 'static']:
            actor_path = self.actors_dir / subdir / f"{actor_name}.json"
            if actor_path.exists():
                try:
                    with open(actor_path, 'r') as f:
                        actor_data = json.load(f)

                    actor_def = ActorDefinition(actor_data, str(actor_path))

                    # Cache the actor
                    self._actor_cache[actor_name] = actor_def

                    return actor_def
                except Exception as e:
                    print(f"Error loading actor '{actor_name}' from {actor_path}: {e}")
                    return None

        print(f"Actor '{actor_name}' not found in actor library")
        return None


    def load_scene(self, scene_name: str, scene_type: str = 'auto') -> Optional[SceneDefinition]:
        """
        Load a scene by name.

        Args:
            scene_name: Name of the scene (without .json extension)
            scene_type: 'training', 'testing', or 'auto' (searches both)

        Returns:
            SceneDefinition object or None if not found
        """
        # Try to find the scene file
        scene_path = None

        if scene_type == 'auto':
            # Search in both training and testing directories
            for directory in [self.scenes_dir / 'training', self.scenes_dir / 'testing']:
                potential_paths = [
                    directory / f"{scene_name}.json",
                ]
                # Also try with numeric prefix (e.g., 01_empty_road.json)
                for file in directory.glob("*.json"):
                    if file.stem.endswith(scene_name) or file.stem == scene_name:
                        potential_paths.append(file)

                for path in potential_paths:
                    if path.exists():
                        scene_path = path
                        break

                if scene_path:
                    break
        else:
            # Search in specific directory
            directory = self.scenes_dir / scene_type
            scene_path = directory / f"{scene_name}.json"
            if not scene_path.exists():
                # Try with numeric prefix
                for file in directory.glob("*.json"):
                    if file.stem.endswith(scene_name) or file.stem == scene_name:
                        scene_path = file
                        break

        if not scene_path or not scene_path.exists():
            print(f"Scene '{scene_name}' not found")
            return None

        # Load scene JSON
        try:
            with open(scene_path, 'r') as f:
                scene_data = json.load(f)
        except Exception as e:
            print(f"Error loading scene from {scene_path}: {e}")
            return None

        # Load actors referenced in the scene
        actor_names = scene_data.get('actors', [])
        actors = []

        for actor_name in actor_names:
            actor = self.load_actor(actor_name)
            if actor:
                actors.append(actor)
            else:
                print(f"Warning: Actor '{actor_name}' referenced in scene '{scene_name}' not found")

        # Create scene definition
        scene_def = SceneDefinition(scene_data, str(scene_path), actors)

        # Validate
        is_valid, error_msg = scene_def.validate()
        if not is_valid:
            print(f"Scene validation failed: {error_msg}")
            return None

        return scene_def

    def list_scenes(self, scene_type: str = 'all') -> Dict[str, List[str]]:
        """
        List all available scenes.

        Args:
            scene_type: 'training', 'testing', or 'all'

        Returns:
            Dictionary with scene types as keys and lists of scene names as values
        """
        scenes = {}

        if scene_type in ['all', 'training']:
            training_dir = self.scenes_dir / 'training'
            if training_dir.exists():
                scenes['training'] = sorted([f.stem for f in training_dir.glob('*.json')])

        if scene_type in ['all', 'testing']:
            testing_dir = self.scenes_dir / 'testing'
            if testing_dir.exists():
                scenes['testing'] = sorted([f.stem for f in testing_dir.glob('*.json')])

        return scenes

    def list_actors(self) -> Dict[str, List[str]]:
        """
        List all available actors in the actor library.

        Returns:
            Dictionary with actor types as keys and lists of actor names as values
        """
        actors = {}

        for subdir in ['vehicles', 'pedestrians', 'static']:
            actor_dir = self.actors_dir / subdir
            if actor_dir.exists():
                actors[subdir] = sorted([f.stem for f in actor_dir.glob('*.json')])

        return actors
