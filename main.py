from orquestrador import AgenteOrquestrador


def main():
    agente = AgenteOrquestrador()

    pergunta = "Gostaria que você me explicasse como funcionam os desviios condicionais"
    resposta = agente.consultar(pergunta)

    print(resposta)


if __name__ == "__main__":
    main()
