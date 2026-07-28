from typing import List

from pydantic import BaseModel, Field


class DetalhesImagemModelo(BaseModel):
    titulo: str = Field(description="Defina sempre o titulo adequado para a imagem que foi analisada")
    descricao: str = Field(description="Coloque aqui uma descrição detalhada de sua analise para imagem")
    rotulos: List[str] = Field(description="Defina tres rótulos principais para a imagem analisar")
