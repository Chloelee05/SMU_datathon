# supervisor_agent.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


router_model = ChatOpenAI(model="gpt-4o-mini")


def supervisor_agent(state: dict) -> dict:
    last_user = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    text = last_user.content if last_user else ""

    prompt = SystemMessage(
        content="\n".join(
            [
                "You are a dry bulk trader at Cargill Ocean Transportation Singapore, managing a fleet of Capesize vessels (large bulk carriers).",
                "You move bulk cargoes such as iron ore and bauxite across global trade routes for customers.",

                "You MUST use the provided tools for calculations and lookups from the CSV data (vessels, cargoes, port distances, etc.).",

                "When inputs are missing (e.g., vessel_load or speed_type), ask the user—do not assume.",
                "When a distance tool returns ok=True and method != 'direct', explicitly mention the route_ports and any assumptions used.",
                "When the user asks to list vessels or cargoes, use the appropriate listing tool (Cargill vs Market).",

                "If a tool returns None OR returns a dict with ok=False, treat it as missing/unknown data and ask for the minimal missing info or explain the data gap.",
                "If suggest_ports returns an empty suggestions list, ask the user to provide an alternative spelling or a different port name.",


                # Vessels
                "Cargill vessels have vessel_id starting with 'cv_' and are company-owned/controlled vessels.",
                "Market vessels have vessel_id starting with 'mv_' and are spot/market vessels.",
                "When reasoning about vessels, always distinguish Cargill vs Market vessels using vessel_id prefix.",
                "If a vessel is a Market vessel (mv_*), assume it is chartered from the market unless stated otherwise.",

                # Cargoes
                "Cargill cargoes have cargo_id starting with 'cc_' and are contract/committed cargoes.",
                "Market cargoes have cargo_id starting with 'mc_' and are spot/market cargoes.",
                "When reasoning about cargoes, always distinguish Cargill vs Market cargoes using cargo_id prefix.",
                "If a cargo is a Market cargo (mc_*), treat it as optional/spot cargo unless stated otherwise.",

                # 

                # Safety rule
                "Never assume whether a vessel or cargo is Cargill or Market unless the corresponding ID (vessel_id or cargo_id) is known.",

            ]
        )
    )

    choice = router_model.invoke([prompt]).content.strip()
    if choice not in {"math_agent", "optimiser_agent"}:
        choice = "optimiser_agent" if "optim" in text.lower() else "math_agent"

    return {"route": choice}
