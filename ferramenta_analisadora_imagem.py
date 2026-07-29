from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from detalhes_imagem_modelo import DetalhesImagemModelo
from my_helper import encode_image
from my_keys import GEMINI_API_KEY
from my_models import GEMINI_LITE


@tool(return_direct=True)
def ferramenta_analisadora_imagem(nome_imagem: str) -> dict:
    """
    Utilize esta ferramenta sempre que for solicitado que você faça uma análise
    de imagem.

    Args:
        nome_imagem (str): Nome da imagem a ser analisada com extensão (ex: teste.jpg).
    """
    llm_lite = ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, model=GEMINI_LITE)

    imagem = encode_image(rf"dados/{nome_imagem}")

    template = ChatPromptTemplate.from_messages([
        ("system", """Assuma que você é um analisador de imagens. 
A sua tarefa principal consiste em: analisar uma imagem e extrair informações importantes e de forma objetiva.

# FORMATO DE SAIDA
Descrição da Imagem: Coloque a sua descrição da imagem.
Rótulos: Coloque uma lista com tres termos chave separados por virgula"""),
        ("user", [
            {"type": "text", "text": "Descreva a imagem: "},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{imagem_informada}"}}
        ])
    ])

    cadeia_analise_imagem = template | llm_lite | StrOutputParser()
    parser_json_imagem = JsonOutputParser(pydantic_object=DetalhesImagemModelo)

    template_resposta = PromptTemplate(
        template="""Gere um resumo utilizando uma linguagem clara e objetiva focada no publico brasileiro. 
A ideia da comunicação do resultado seja o mais fácil possível, priorizando registros para consultas posteriors.

# O Resultado da imagem
{resposta_cadeia_analise_imagem}

# FORMATO DE SAIDA
{formato_saida}""",
        input_variables=["resposta_cadeia_analise_imagem"],
        partial_variables={"formato_saida": parser_json_imagem.get_format_instructions()}
    )

    # Correção do pipeline: mapeia a string gerada pela primeira cadeia para o dicionário da segunda
    cadeia_completa = (
            cadeia_analise_imagem
            | (lambda x: {"resposta_cadeia_analise_imagem": x,
                          "formato_saida": parser_json_imagem.get_format_instructions()})
            | template_resposta
            | llm_lite
            | parser_json_imagem
    )

    resposta = cadeia_completa.invoke({"imagem_informada": imagem})
    return resposta
