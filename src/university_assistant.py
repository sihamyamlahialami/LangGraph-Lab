"""LangGraph University Assistant — five agents + three tools, Groq backend.

Usage:
    python university_assistant.py --self-test  # SQLite/calculator; no Groq key
    python university_assistant.py --demo       # Calls Groq (API key required)
    python university_assistant.py              # Interactive chat
"""

import argparse
import ast
from datetime import datetime
from getpass import getpass
import operator as op
import os
from pathlib import Path
import sqlite3
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()
DB_PATH = Path(__file__).resolve().parent / 'university_courses_demo.db'
CATEGORIES = ['COURSES','PROGRAMMING','MATHEMATICS','ADMINISTRATION','GENERAL']
NODE_NAMES = {c: f'{c.lower()}_agent' for c in CATEGORIES}

DEMO_COURSES = [
    ('ML101','Machine Learning','Supervised learning, unsupervised learning, model evaluation.',6,'Python, Statistics, Linear Algebra'),
    ('DL201','Deep Learning','Neural networks, CNNs, RNNs, Transformers.',6,'Machine Learning, Python'),
    ('PY101','Python Programming','Python syntax, functions, OOP, data structures.',4,'None'),
    ('DB101','Database Systems','SQL, relational databases, normalization, transactions.',5,'Basic Programming'),
    ('MA101','Linear Algebra','Vectors, matrices, eigenvalues, linear transformations.',5,'Basic Mathematics'),
]


def initialize_database(path=DB_PATH):
    """Create/reset a small fictional course catalog for educational use."""
    with sqlite3.connect(path) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS courses (
            code TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT,
            credits INTEGER,
            prerequisites TEXT
        )''')
        conn.executemany('''INSERT OR REPLACE INTO courses
            (code,title,description,credits,prerequisites)
            VALUES(?,?,?,?,?)''', DEMO_COURSES)
        conn.commit()


OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod, ast.Pow: op.pow,
    ast.USub: op.neg, ast.UAdd: op.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int,float):
        if abs(node.value) > 1e9:
            raise ValueError('Number too large')
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        a, b = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(b)>12:
            raise ValueError('Exponent too large')
        result=OPS[type(node.op)](a,b)
        if not isinstance(result,(int,float)) or abs(result)>1e15:
            raise ValueError('Result too large')
        return result
    raise ValueError('Unsupported expression')


@tool
def calculator(expression: str) -> str:
    """Calculate basic arithmetic expressions, such as '(15 + 27) * 3'."""
    if len(expression)>100:
        return 'Invalid expression'
    try:
        return str(_safe_eval(ast.parse(expression.strip(), mode='eval')))
    except (ValueError, ArithmeticError, SyntaxError, TypeError, OverflowError):
        return 'Invalid expression'


@tool
def current_datetime(_: str = '') -> str:
    """Get the current system date and time. Use for current-date/time questions."""
    return datetime.now().strftime('%A, %d %B %Y, %H:%M:%S')


@tool
def search_courses(course_name: str) -> str:
    """Look up fictional demo university courses by course name or code, including credits and prerequisites."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows=conn.execute('''SELECT code,title,description,credits,prerequisites
                FROM courses
                WHERE LOWER(title) LIKE LOWER(?) OR LOWER(code) LIKE LOWER(?)
                ORDER BY code''', (f'%{course_name}%',f'%{course_name}%')).fetchall()
    except sqlite3.Error as exc:
        return f'Database error: {type(exc).__name__}'
    if not rows:
        return 'No matching course found in the fictional demo catalog.'
    return '\n\n'.join(
        f'Course: {title}\nCode: {code}\nDescription: {description}\n'
        f'Credits: {credits}\nPrerequisites: {prerequisites}'
        for code,title,description,credits,prerequisites in rows
    )


class AssistantState(TypedDict):
    messages: Annotated[list,add_messages]
    category: str


SYSTEM_PROMPTS={
    'COURSES': (
        'You are a university courses advisor. Use search_courses when the '
        'student asks for credits or prerequisites in the demo course catalog. '
        'This catalog is fictional, not official. Do not invent schedules or school-specific rules. '
        'Explain briefly, and use the tool result as your evidence. After receiving a tool result, answer.'
    ),
    'PROGRAMMING': 'You are a university Python programming tutor. Provide concise and correct code when needed.',
    'MATHEMATICS': (
        'You are a mathematics tutor. Show concise steps and ALWAYS use the calculator tool '
        'for arithmetic. After receiving the tool result, answer.'
    ),
    'ADMINISTRATION': (
        'You are a university administration assistant. Give general guidance only. '
        'Do not invent official contacts, fees, procedures or deadlines.'
    ),
    'GENERAL': (
        'You are a concise university assistant. Use current_datetime for questions about '
        'the current date and time. After receiving the tool result, answer.'
    ),
}


def build_assistant():
    """Construct and compile LangGraph with tool calls returning to the requesting agent."""
    initialize_database()
    if not os.environ.get('GROQ_API_KEY'):
        os.environ['GROQ_API_KEY']=getpass('Groq API key (hidden): ')

    llm=ChatGroq(model='openai/gpt-oss-20b',temperature=0,
                 reasoning_effort='low',max_tokens=2048)
    tools=[calculator,current_datetime,search_courses]
    llm_with_tools=llm.bind_tools(tools)  # Must come AFTER tools list is complete.

    def classify(state:AssistantState):
        text=str(state['messages'][-1].content).strip()
        if not text:
            return {'category':'EMPTY'}
        prompt=(
            'Classify the most recent university student question as exactly one label: '
            'COURSES (course content, course prerequisites, course credits, syllabus, exams); '
            'PROGRAMMING (code); MATHEMATICS (math calculations); '
            'ADMINISTRATION (enrollment, certificates, forms, fees); GENERAL (other). '
            'Return only the label.\nQuestion: '+text
        )
        try:
            raw=str(llm.invoke(prompt).content).strip().upper()
            label=next((c for c in CATEGORIES if raw==c),'GENERAL')
            return {'category':label}
        except Exception as exc:
            print('Classifier API error:',type(exc).__name__)
            return {'category':'ERROR'}

    def make_agent(category):
        system=SystemMessage(content=SYSTEM_PROMPTS[category])
        def agent(state:AssistantState):
            try:
                return {'messages':[llm_with_tools.invoke([system]+state['messages'])]}
            except Exception as exc:
                return {'messages':[AIMessage(content=f'Technical error: {type(exc).__name__}. Please try again.')]}
        return agent

    def fallback(state:AssistantState):
        if state['category']=='EMPTY':
            msg='Please enter a question.'
        else:
            msg='Sorry, the question could not be processed. Please try again.'
        return {'messages':[AIMessage(content=msg)]}

    def route_after_classifier(state:AssistantState):
        return NODE_NAMES.get(state['category'],'fallback')

    def route_after_agent(state:AssistantState):
        return 'tools' if getattr(state['messages'][-1],'tool_calls',None) else END

    def route_after_tools(state:AssistantState):
        return NODE_NAMES.get(state['category'],'general_agent')

    builder=StateGraph(AssistantState)
    builder.add_node('classifier',classify)
    builder.add_node('fallback',fallback)
    builder.add_node('tools',ToolNode(tools,handle_tool_errors=True))
    for category in CATEGORIES:
        builder.add_node(NODE_NAMES[category],make_agent(category))
    builder.add_edge(START,'classifier')
    builder.add_conditional_edges(
        'classifier',route_after_classifier,
        {**{name:name for name in NODE_NAMES.values()},'fallback':'fallback'}
    )
    for category in CATEGORIES:
        builder.add_conditional_edges(NODE_NAMES[category],route_after_agent,
                                      {'tools':'tools',END:END})
    builder.add_conditional_edges('tools',route_after_tools,
                                  {name:name for name in NODE_NAMES.values()})
    builder.add_edge('fallback',END)
    return builder.compile(checkpointer=MemorySaver())


def ask(assistant,question,thread_id='demo'):
    result=assistant.invoke(
        {'messages':[HumanMessage(content=question)]},
        {'configurable':{'thread_id':thread_id}}
    )
    print('\nQUESTION:',repr(question),'\nCATEGORY:',result['category'])
    calls=[call['name'] for m in result['messages']
           for call in getattr(m,'tool_calls',[])]
    print('TOOLS CALLED:',calls)
    for message in result['messages']:
        if isinstance(message,ToolMessage):
            print('TOOL OUTPUT:',message.name,str(message.content)[:500])
    print('FINAL ANSWER:',result['messages'][-1].content)
    return result


def self_test():
    initialize_database()
    r=calculator.invoke({'expression':'(15 + 27) * 3'})
    assert r=='126',r
    r=search_courses.invoke({'course_name':'Machine Learning'})
    assert 'ML101' in r and 'Python, Statistics, Linear Algebra' in r,r
    assert 'Invalid expression'==calculator.invoke({'expression':'__import__("os")'})
    assert len(DEMO_COURSES)==5
    print('OFFLINE SELF-TEST: PASS (calculator and SQLite)')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true',help='No API key required')
    parser.add_argument('--demo',action='store_true',help='Run Groq demo prompts')
    args=parser.parse_args()
    if args.self_test:
        self_test()
        return
    assistant=build_assistant()
    if args.demo:
        tests=[
            ('What topics are usually covered in a machine learning course?','course'),
            ('Write a Python function that computes factorial(n).','programming'),
            ('Calculate (15 + 27) * 3 using the calculator.','math'),
            ('How do I request an enrollment certificate?','admin'),
            ('Use the current_datetime tool and tell me the current date.','date'),
            ('Using the demo database, what are Machine Learning prerequisites? Use search_courses.','sqlite'),
            ('   ','empty'),
        ]
        for question,thread in tests:
            ask(assistant,question,thread)
        ask(assistant,'My name is Ahmed.','memory')
        ask(assistant,'What is my name?','memory')
        return
    print('University assistant — type exit to stop.')
    while True:
        try:
            question=input('You: ')
        except (EOFError,KeyboardInterrupt):
            break
        if question.strip().lower() in {'exit','quit'}:
            break
        ask(assistant,question,thread_id='interactive')


if __name__=='__main__':
    main()
