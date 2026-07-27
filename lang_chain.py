from langchain_core.globals import set_debug
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from my_helper import encode_image
from my_keys import GEMINI_API_KEY
from my_models import GEMINI_FLASH, GEMINI_LITE

set_debug(True)

llm_flash = ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, model=GEMINI_FLASH)

imagem = encode_image("dados/exemplo_grafico.jpg")

template = ChatPromptTemplate.from_messages([("system",
                                              """Assuma que você é um analisador de imagens. 
                                              A sua tarefa principal consiste em: analisar uma imagem 
                                              e extrair informações importantes e de forma objetiva.
                                              
                                              # FORMATO DE SAIDA
                                              Descrição da Imagem: 'Coloque a sua descrição da imagem.
                                              Rótulos: "Coloque uma lista com tres termos chave separados por virgula'
                                              """),
                                             ("user",
                                              [{"type": "text",
                                                "text": "Descreva a imagem: "},
                                               {"type": "image_url",
                                                "image_url": {"url": "data:image/jpeg;base64,{imagem_informada}"}}])])

cadeia_analise_imagem = template | llm_flash | StrOutputParser()

template_resposta = PromptTemplate(template="""Gere um resumo utilizando uma linguagem clara e objetiva focada no 
                                            publico brasileiro. A ideia da comunicação do resultado seja o mais 
                                            fácil possível, priorizando registros para consultas posteriors.
                                            
                                            #O Resultado da imagem
                                            {resposta_cadeia_analise_imagem}""",
                                   input_variables=["resposta_cadeia_analise_imagem"])

llm_lite = ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, model=GEMINI_LITE)

cadeia_resumo = template_resposta | llm_lite | StrOutputParser()

cadeia_completa = (cadeia_analise_imagem | cadeia_resumo)

resposta = cadeia_completa.invoke({"imagem_informada": imagem})

print(resposta)
