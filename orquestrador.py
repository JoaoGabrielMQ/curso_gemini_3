from langchain.agents import create_agent
from langchain_core.globals import set_debug
from langchain_google_genai import ChatGoogleGenerativeAI

from ferramenta_analisadora_imagem import ferramenta_analisadora_imagem
from ferramente_explicadora import ferramente_explicador
from detalhes_imagem_modelo import DetalhesImagemModelo
from my_keys import GEMINI_API_KEY
from my_models import GEMINI_LITE

set_debug(False)


class AgenteOrquestrador:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, model=GEMINI_LITE)
        self.tools = [ferramenta_analisadora_imagem, ferramente_explicador]
        self.agente = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt="Você é um agente pessoal focado em ajudar com tarefas de trabalho e estudos. "
                          "Dê respostas em português de forma simples e direta.",
        )

    def consultar(self, pergunta: str):
        resposta = self.agente.invoke({"messages": [("user", pergunta)]})

        historico_mensagens = resposta.get("messages", [])
        if historico_mensagens:
            ultima_mensagem = historico_mensagens[-1]
            return ultima_mensagem.content

        return "Não foi possível obter nenhuma resposta."
