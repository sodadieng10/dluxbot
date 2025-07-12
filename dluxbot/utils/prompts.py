def build_prompt(user_input, memory):
    return f"""
Tu es Dluxbot, un assistant intelligent. Voici le contexte : {memory}
Utilisateur : {user_input}
Dluxbot :
"""
