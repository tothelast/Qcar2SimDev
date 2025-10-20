"""Scene and actor loader from JSON files."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any


class ActorDefinition:
    """Actor definition loaded from JSON."""

    def __init__(self, actor_data: Dict[str, Any], actor_path: str):
        """Initialize actor from JSON data."""
        self.actor_path = actor_path
        self.name = actor_data.get('name', 'unnamed_actor')
        self.description = actor_data.get('description', '')
        self.type = actor_data.get('type', 'unknown')
        self.actor_number = actor_data.get('actor_number', 0)
        self.data = actor_data

    def __repr__(self):
        return f"ActorDefinition(name='{self.name}', type='{self.type}', actor_number={self.actor_number})"


class SceneDefinition:
    """Scene definition with route and actors."""

    def __init__(self, scene_data: Dict[str, Any], scene_path: str, actors: List[ActorDefinition]):
        """Initialize scene from JSON data."""
        self.scene_path = scene_path
        self.name = scene_data.get('name', 'unnamed_scene')
        self.description = scene_data.get('description', '')
        self.ego_route = scene_data.get('ego_route', '')
        self.actors = actors

        # Categorize actors by type
        self.autonomous_vehicles = [a for a in actors if a.type == 'autonomous_vehicle']
        self.pedestrians = [a for a in actors if a.type == 'pedestrian']
        self.parked_vehicles = [a for a in actors if a.type == 'parked_vehicle']
        self.stop_signs = [a for a in actors if a.type == 'stop_sign']

    def validate(self) -> tuple[bool, str]:
        """Validate scene has required fields."""
        if not self.ego_route:
            return False, f"Scene '{self.name}' missing ego_route"
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
    """Loads scenes and actors from JSON files."""

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize loader with config directory."""
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / 'config'

        self.config_dir = Path(config_dir)
        self.actors_dir = self.config_dir / 'actors'
        self.scenes_dir = self.config_dir / 'scenes'
        self.routes_dir = self.config_dir / 'routes'
        self._actor_cache: Dict[str, ActorDefinition] = {}

    def load_actor(self, actor_name: str) -> Optional[ActorDefinition]:
        """Load actor from cache or actor library."""
        if actor_name in self._actor_cache:
            return self._actor_cache[actor_name]

        for subdir in ['vehicles', 'pedestrians', 'static']:
            actor_path = self.actors_dir / subdir / f"{actor_name}.json"
            if actor_path.exists():
                try:
                    with open(actor_path, 'r') as f:
                        actor_def = ActorDefinition(json.load(f), str(actor_path))
                    self._actor_cache[actor_name] = actor_def
                    return actor_def
                except Exception as e:
                    print(f"Error loading actor '{actor_name}': {e}")
                    return None

        print(f"Actor '{actor_name}' not found")
        return None


    def load_scene(self, scene_name: str, scene_type: str = 'auto') -> Optional[SceneDefinition]:
        """Load scene by name from training/testing directories."""
        scene_path = self._find_scene_path(scene_name, scene_type)
        if not scene_path:
            print(f"Scene '{scene_name}' not found")
            return None

        try:
            with open(scene_path, 'r') as f:
                scene_data = json.load(f)
        except Exception as e:
            print(f"Error loading scene: {e}")
            return None

        # Load referenced actors
        actors = []
        for actor_name in scene_data.get('actors', []):
            actor = self.load_actor(actor_name)
            if actor:
                actors.append(actor)
            else:
                print(f"Warning: Actor '{actor_name}' not found")

        scene_def = SceneDefinition(scene_data, str(scene_path), actors)
        is_valid, error_msg = scene_def.validate()
        if not is_valid:
            print(f"Scene validation failed: {error_msg}")
            return None

        return scene_def

    def _find_scene_path(self, scene_name: str, scene_type: str) -> Optional[Path]:
        """Find scene file in appropriate directory."""
        if scene_type == 'auto':
            for directory in [self.scenes_dir / 'training', self.scenes_dir / 'testing']:
                path = self._search_scene_in_dir(directory, scene_name)
                if path:
                    return path
        else:
            return self._search_scene_in_dir(self.scenes_dir / scene_type, scene_name)
        return None

    def _search_scene_in_dir(self, directory: Path, scene_name: str) -> Optional[Path]:
        """Search for scene file in directory."""
        path = directory / f"{scene_name}.json"
        if path.exists():
            return path
        for file in directory.glob("*.json"):
            if file.stem == scene_name or file.stem.endswith(scene_name):
                return file
        return None

    def list_scenes(self, scene_type: str = 'all') -> Dict[str, List[str]]:
        """List available scenes by type."""
        scenes = {}
        for stype in (['training', 'testing'] if scene_type == 'all' else [scene_type]):
            sdir = self.scenes_dir / stype
            if sdir.exists():
                scenes[stype] = sorted([f.stem for f in sdir.glob('*.json')])
        return scenes

    def list_actors(self) -> Dict[str, List[str]]:
        """List available actors by type."""
        actors = {}
        for subdir in ['vehicles', 'pedestrians', 'static']:
            adir = self.actors_dir / subdir
            if adir.exists():
                actors[subdir] = sorted([f.stem for f in adir.glob('*.json')])
        return actors
