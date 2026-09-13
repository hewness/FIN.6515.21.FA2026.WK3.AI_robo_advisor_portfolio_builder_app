from portfolio_builder.ui import CSS, HEAD, THEME, build_demo

demo = build_demo()

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, head=HEAD)
