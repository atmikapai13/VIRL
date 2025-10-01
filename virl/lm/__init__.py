from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
from langchain_community.chat_models import ChatOpenAI

from virl.config import cfg
from virl.utils.common_utils import print_prompt, print_answer, parse_answer_to_json
from .gpt_chat import GPTChat

# Guard Azure import so non-Azure users don't fail at import time
try:
    from .azure_gpt import AzureGPTChat  # requires Azure credentials at import
except Exception:
    AzureGPTChat = None


__all__ = {
    'GPT': GPTChat,
}
if AzureGPTChat is not None:
    __all__['AzureGPT'] = AzureGPTChat


def build_chatbot(name):
    if name not in __all__:
        raise KeyError(f"Chatbot '{name}' is not available. Ensure required deps/credentials are set.")
    return __all__[name](cfg)


class UnifiedChat(object):
    chatbots = None

    def __init__(self):
        UnifiedChat.chatbots = {
            name: build_chatbot(name) for name in cfg.LLM.NAMES
        }

    @classmethod
    def ask(cls, question, **kwargs):
        print_prompt(question)
        chatbot = kwargs.get('chatbot', cfg.LLM.DEFAULT)
        answer = cls.chatbots[chatbot].ask(question, **kwargs)
        print_answer(answer)

        return answer

    @classmethod
    def search(cls, question, json=False):
        llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
        tools = load_tools(["serpapi"], llm=llm)
        agent = initialize_agent(tools, llm, verbose=True)
        answer = agent.run(question)

        if json:
            answer = parse_answer_to_json(answer)

        return answer
