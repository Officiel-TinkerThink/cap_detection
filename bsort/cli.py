import typer

app = typer.Typer()

@app.command()
def train():
    print('You are going training')


@app.command()
def infer():
    print("Your are going inference")

def main():
  app()

if __name__ == "__main__":
    main()