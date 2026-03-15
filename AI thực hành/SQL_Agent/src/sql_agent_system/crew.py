# src/sql_agent_system/crew.py

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, task, crew
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.task_output import TaskOutput
from .tools.database_schema_tool import DatabaseSchemaTool
from .tools.sql_execution_tool import SQLExecutionTool
from .tools.knowledge_tool import create_knowledge_tool  
import os
from dotenv import load_dotenv

load_dotenv()

# Bypass CrewAI's OpenAI key validation
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "dummy-key-for-validation-only"


# Condition function for ConditionalTask
def needs_sql_execution(output: TaskOutput) -> bool:
	"""
	Determine if SQL execution is needed based on sql_translator's output.
	Returns True if SQL query was generated, False if metadata/knowledge answer was provided.
	
	The sql_translator should indicate in its output whether a query is needed.
	If the output contains SQL keywords like SELECT, it means execution is needed.
	If the output is just informational (from knowledge base or schema info), skip execution.
	"""
	# Use the 'raw' attribute which contains the actual output text
	result_text = str(output.raw).lower() if output.raw else ""
	
	# Keywords that indicate a SQL query was generated
	sql_keywords = ['select ', 'from ', 'where ', 'join ', 'group by', 'order by']
	
	# If it looks like a SQL query, it needs execution
	has_sql = any(keyword in result_text for keyword in sql_keywords)
	
	# If the answer is clearly from knowledge base or metadata only, no execution needed
	no_exec_keywords = ['here is the information', 'according to', 'from our knowledge base', 
	                      'person in charge', 'responsible for', 'metadata', 'schema information']
	is_metadata_answer = any(keyword in result_text for keyword in no_exec_keywords)
	
	return has_sql and not is_metadata_answer


@CrewBase
class SqlAgentSystemCrew():
	"""SqlAgentSystem crew with custom knowledge system"""
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	def __init__(self):
		# ✅ Initialize custom knowledge tool once
		self._knowledge_tool = None

	def llm(self):
		return LLM(
					model=os.getenv("MODEL_NAME"),
					api_key=os.getenv("GEMINI_API_KEY"),
					temperature=0.0  # Lower temperature for more consistent results.
				)
	
	@property
	def knowledge_tool(self):
		"""Lazy load knowledge tool (initialized once)"""
		if self._knowledge_tool is None:
            # Current file: src/sql_agent_system/crew.py
            # Knowledge file: knowledge/forecast_error_metrics.txt (Project Root)
            
            # Go up two levels: src/sql_agent_system -> src -> root
			project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
			knowledge_path = os.path.join(
				project_root,
				"knowledge",
				"forecast_error_metrics.txt"
			)
			self._knowledge_tool = create_knowledge_tool(knowledge_path)
		return self._knowledge_tool

	@agent
	def sql_translator(self) -> Agent:
		return Agent(
			config=self.agents_config['sql_translator'],
			tools=[
				DatabaseSchemaTool(),
				self.knowledge_tool  # ✅ Add knowledge tool
			],
			llm=self.llm(),
			verbose=True
		)

	@agent
	def sql_executor(self) -> Agent:
		return Agent(
			config=self.agents_config['sql_executor'],
			tools=[
				SQLExecutionTool(),
				self.knowledge_tool  # ✅ Add knowledge tool
			],
			llm=self.llm(),
			verbose=True
		)

	@task
	def translate_task(self) -> Task:
		return Task(
			config=self.tasks_config['translate_task'],
		)

	@task
	def execute_task(self) -> ConditionalTask:
		"""
		Conditional task: Only executes if sql_translator generated a SQL query.
		If the question was answered using metadata/knowledge, this task is skipped.
		"""
		return ConditionalTask(
			config=self.tasks_config['execute_task'],
			condition=needs_sql_execution,
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the SqlAgentSystem crew with custom knowledge"""
		return Crew(
			agents=self.agents,  # Automatically created by @agent decorator
			tasks=self.tasks,    # Automatically created by @task decorator
			process=Process.sequential,
			verbose=True,
		)