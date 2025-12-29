from langgraph.graph import StateGraph,END,START
from typing import TypedDict
from langchain_ollama import ChatOllama


class BMIstate(TypedDict):
    weight: float
    height: float
    bmi: float
    categorey: str




def calculate_bmi(state: BMIstate)->BMIstate:
    weight=state['weight']
    height=state['height']
    bmi=weight/(height**2)

    state['bmi']=round(bmi,2)
    return state

def label_bmi(state:BMIstate)->BMIstate:

    bmi=state['bmi']
    
    if bmi<18.5:
        state['categorey']='underweight'

    elif bmi>=18.5 and bmi<25:
        state['categorey']='Normal'

    elif bmi>=25 and bmi <30:
        state['categorey']='Overweight'
    else:
        state['categorey']='Obese'    
    
    return state


def build_grph(graph):

    graph.add_node('calculate_bmi',calculate_bmi)
    graph.add_node('label_bmi',label_bmi)

    graph.add_edge(START,'calculate_bmi')
    graph.add_edge('calculate_bmi','label_bmi')

    graph.add_edge('label_bmi',END)

    workflow=graph.compile()
    return workflow

def test_llm():

     llm = ChatOllama(model="llama3.2:1b",
                     temperature=0.2,num_predict=256)
     responce=llm.invoke("how far moon from earth")
     print(responce)



if __name__=="__main__":

  initail_graph={'weight':80,'height':1.73}
  graph=StateGraph(BMIstate)
  workflow=build_grph(graph=graph)

  final_state=workflow.invoke(initail_graph)

  print(final_state)
  test_llm()
