from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from my_keys import GEMINI_API_KEY
from my_models import GEMINI_LITE


@tool(return_direct=True)
def ferramente_explicador(tema_parametro: str):
    """
    Utilize essa ferramente sempre que for solicitado que você explique um conteúdo para pessoas.

    # Entrada Requeridas
     - 'tema' (str): Tema principal informado na pergunta do usuário.
    """
    llm_lite = ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, model=GEMINI_LITE)

    template_resposta = PromptTemplate(template="""Assuma o papel de um professor preocupado com os aspectos didáticos do usuário
    1. Elabore uma explicação sobre o tema {tema} que seja compreensível com estudantes na fase de conclusão do ensino medio.
    2. Utilize exemplos do cotidiano para tornar a explicação mais fácil.
    3. Caso sugira algum recurso para apoiar a explicação lembre-se do cenário brasileiro.
    4. Caso voce presenter um código, seja didático e utilize python.
    
    Tema pergunta: {tema}""", input_variables=["tema"])

    cadeia = template_resposta | llm_lite | StrOutputParser()
    resposta = cadeia.invoke({"tema": tema_parametro})

    return resposta
