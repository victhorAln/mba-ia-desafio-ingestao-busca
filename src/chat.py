from search import search_prompt

SAIR = ("sair", "exit", "quit")


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Faça sua pergunta (digite 'sair' para encerrar):")

    while True:
        try:
            pergunta = input("\nPERGUNTA: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando o chat.")
            break

        if not pergunta:
            continue
        if pergunta.lower() in SAIR:
            print("Encerrando o chat.")
            break

        try:
            resposta = chain(pergunta)
        except Exception as e:
            print(f"Erro ao processar a pergunta: {e}")
            continue

        print(f"RESPOSTA: {resposta}")


if __name__ == "__main__":
    main()
