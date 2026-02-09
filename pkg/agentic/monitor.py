from typing import Dict


class AgentMonitor:
    """Agent性能监控"""
    def __init__(self):
        self.metrics = {
            "success_rate": 0,
            "avg_iterations": 0,
            "tool_usage": {},
            "reflection_depth": 0
        }

    def evaluate_agent(self, final_state: Dict) -> Dict:
        """评估Agent表现"""
        return {
            "success": self._check_success(final_state),
            "efficiency": len(final_state["tool_results"]) / final_state["iteration"],
            "reflection_quality": self._analyze_reflections(final_state["reflections"]),
            "tool_effectiveness": self._analyze_tool_usage(final_state["tool_results"])
        }