from portfolio_builder.ui import CSS, THEME, build_demo

demo = build_demo()

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
