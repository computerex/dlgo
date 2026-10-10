"""
Z-Agent Adapter for Harbor Terminal-Bench

This agent wraps the user's 'z' streaming harness tool for Harbor Terminal-Bench evaluation.
"""

import subprocess
import os
import json
from typing import Dict, Any, Optional
from harbor.models.trial.result import AgentInfo, ModelInfo


class ZAgent:
    """
    Harbor agent wrapper for the 'z' streaming harness tool.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Z Agent.
        
        Args:
            **kwargs: Additional configuration options including:
                - workspace: Directory to use for z workspace
                - session: Session name for z
                - model: Model to use with z
                - api_url: API URL for z
                - api_key: API key for z
                - timeout: Command timeout in seconds
        """
        self.config = kwargs
        self.z_command = self.config.get('z_command', 'z')
        self.workspace = self.config.get('workspace')
        self.session = self.config.get('session', 'default')
        self.model = self.config.get('model')
        self.api_url = self.config.get('api_url')
        self.api_key = self.config.get('api_key')
        self.timeout = self.config.get('timeout', 600)
        self.environment = None
    
    async def setup(self, environment):
        """
        Setup the agent with the Harbor environment.
        
        Args:
            environment: The Harbor environment object
        """
        self.environment = environment
        # You can access environment properties here if needed
        # For example: environment.working_directory, etc.
        return None
        
    def build_z_command(self, task_input: str) -> list:
        """
        Build the z command with appropriate parameters.
        
        Args:
            task_input: The task description or input
            
        Returns:
            List of command components
        """
        cmd = [self.z_command]
        
        # Add workspace if specified
        if self.workspace:
            cmd.append(self.workspace)
        
        # Add session parameter
        cmd.extend(['--session', self.session])
        
        # Add model if specified
        if self.model:
            cmd.extend(['--model', self.model])
        
        # Add API URL if specified
        if self.api_url:
            cmd.extend(['--api-url', self.api_url])
            
        # Add API key if specified
        if self.api_key:
            cmd.extend(['--api-key', self.api_key])
        
        # For headless/non-interactive use, we might need additional flags
        # depending on how z is configured
        
        return cmd
    
    def execute(self, task_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute the z command with the given task input.
        
        Args:
            task_input: The task description or input to process
            context: Additional context information about the task
            
        Returns:
            The output from the z command
        """
        cmd = self.build_z_command(task_input)
        
        try:
            # Try to run z and capture output
            # Note: This assumes z can run non-interactively
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=False,  # Capture as bytes to avoid encoding issues
                timeout=self.timeout,
                shell=False
            )
            
            # Decode output with error handling
            stdout = ""
            stderr = ""
            
            try:
                stdout = result.stdout.decode('utf-8', errors='replace')
            except Exception:
                try:
                    stdout = result.stdout.decode('cp1252', errors='replace')
                except Exception:
                    stdout = str(result.stdout)
            
            try:
                stderr = result.stderr.decode('utf-8', errors='replace')
            except Exception:
                try:
                    stderr = result.stderr.decode('cp1252', errors='replace')
                except Exception:
                    stderr = str(result.stderr)
            
            # Return the output
            if result.returncode == 0:
                return stdout or "Z command completed successfully"
            else:
                # If there's an error, return both stdout and stderr
                error_msg = f"Z command failed with exit code {result.returncode}"
                if stdout:
                    error_msg += f"\nStdout:\n{stdout}"
                if stderr:
                    error_msg += f"\nStderr:\n{stderr}"
                return error_msg
                
        except subprocess.TimeoutExpired:
            return f"Error: Z command timed out after {self.timeout} seconds"
        except Exception as e:
            return f"Error executing z command: {str(e)}"
    
    async def run(self, task_description: str = "", **kwargs) -> str:
        """
        Main interface method that Harbor calls.
        
        Args:
            task_description: The task description to process
            **kwargs: Additional task parameters
            
        Returns:
            The agent's response/output
        """
        # Merge any additional kwargs with config
        merged_context = {**self.config, **kwargs}
        
        # Execute the z command
        return self.execute(task_description, context=merged_context)
    
    def to_agent_info(self):
        """
        Return agent information for Harbor.
        
        Returns:
            AgentInfo object containing agent metadata
        """
        # Create model info if a model is specified
        model_info = None
        if self.model:
            model_info = ModelInfo(name=self.model)
        
        return AgentInfo(
            name="z-agent",
            version="1.0",
            model_info=model_info
        )


# For Harbor to discover this agent, we need to export it
__all__ = ['ZAgent']