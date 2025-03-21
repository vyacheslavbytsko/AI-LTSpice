import base64
import io

from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool
from telebot import TeleBot

from misc import get_default_mapping, round16, get_pin_positions


def description_to_simple_circuits_descriptions_tool(llm, known_circuits):
    def process_description(query, known_circuits):
        messages = [
            HumanMessage(
                "Разбей данное описание схемы на более простые схемы, "
                "которые я знаю. Ответ представь в виде списка названий. "
                "То есть, семантически для каждого подописания внутри "
                "большого описания соответствует ОДНА схема. "
                "Если ты видишь, что для какого-то подописания "
                "подходит несколько знакомых мне схем, выбери ту, "
                "которая подходит больше всего."
                f"\n\nСхемы, которые я знаю:\n{known_circuits}\n\nОписание: {query}"
            )
        ]

        response = llm.invoke(messages)

        return response.content

    return Tool(
        name="description_to_simple_circuits",
        description="Разбивает описание сложной схемы на несколько простых описаний схем для последующего поиска.",
        func=lambda query: process_description(query, known_circuits)
    )


def simple_circuit_description_to_filename_tool(vector_store: FAISS):
    def get_relevant_filenames_and_descriptions(query):
        retriever = vector_store.as_retriever(
            search_type="similarity",
            k=3,
            score_threshold=None,
        )

        relevant_descriptions = retriever.invoke(query)

        relevant_descriptions_text = ""
        for relevant_description in relevant_descriptions:
            relevant_descriptions_text += f"Filename \"{relevant_description.metadata["description_filename"]}\": \"{relevant_description.page_content}\"\n\n"

        return f"Here are three filenames and descriptions which fit the best to the description you provided:\n\n{relevant_descriptions_text}Use them to choose filename to get netlist of the circuit. Be aware - those parts were found by using vector store, so if you don't like my answer, reask user for more specific description. DO NOT make user choose filename. Choose by yourself. Choose by yourself."

    return Tool(
        name="description_to_filenames",
        description="Searches and returns filenames and incomplete descriptions of circuits which fits best to the description provided by human to then get netlist using another tool.",
        func=lambda query: get_relevant_filenames_and_descriptions(query)
    )


def filename_to_full_circuit_description_tool():
    def get_content(query):
        return f"Here is the full description of circuit:\n\n\"{open(query).read()}\""

    return Tool(
        name="filename_to_full_circuit_description",
        description="Returns the full description of circuit based on the filename of description.",
        func=lambda query: get_content(query)
    )


def filename_to_netlist_b64_tool():
    def filename_to_netlist_b64(query):
        return (f"Here's the base64 representation of netlist:\n\n```"
                f"{base64.b64encode(
                    open(query.removesuffix(".desc.txt") + ".net")
                    .read()
                    .encode("utf-8")
                ).decode("utf-8")}```\n\nYou can now convert it to the .asc file or send it to the user.")

    return Tool(
        name="filename_to_netlist_b64",
        description="Returns the base64 representation of netlist of circuit based on the filename of description.",
        func=lambda query: filename_to_netlist_b64(query)
    )

def combine_netlists_tool(llm):
    def combine_netlists(netlists: list[str], description: str) -> str:
        netlists_str = ""

        for i, netlist in enumerate(netlists, start=1):
            netlists_str += f"\n* Netlist {i}:\n{netlist}\n"

        prompt = (
            "Объедини следующие netlist'ы в один, основываясь на их содержимом и общем описании. "
            "Убедись, что все компоненты подключены корректно и схема соответствует описанию.\n\n"
            f"Описание схемы: {description}\n\nNetlist'ы: {netlists_str}\n"
            "Верни только итоговый netlist без пояснений."
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content

    return Tool(
        name="combine_netlists",
        description="Объединяет несколько netlist'ов в один общий netlist на основе их содержимого и описания схемы.\n"
                    "\n"
                    "Args:\n"
                    "    netlists (list[str]): A list of netlists' contents (full contents!!!)\n"
                    "    description (str): Initial description of circuit user wants.\n"
                    "\n"
                    "Returns:\n"
                    "    str: Netlist which is a combination of small netlists.",
        func=lambda netlists, description: combine_netlists(netlists, description)
    )


def netlist_b64_to_asc_tool(sheet_size: str = "SHEET 1 800 600",
                        directive_text_coords: str = "TEXT 450 300 Left 2",
                        comment_text_default: str = "TEXT 80 -250 Left 4"):
    def net_to_asc(net_content: str,
                   sheet_size: str = "SHEET 1 800 600",
                   directive_text_coords: str = "TEXT 450 300 Left 2",
                   comment_text_default: str = "TEXT 80 -250 Left 4") -> str:
        """
        Универсально преобразует содержимое .net файла в .asc файл.

        Для каждого описания компонента:
          • Определяется mapping по имени (см. get_default_mapping).
          • Компоненты располагаются на сетке с начальным положением (comp_start_x, comp_start_y) и шагом, кратным 16.
          • Пины компонента вычисляются относительно его центра, координаты округляются до кратных 16.
          • Узлы (флаги) собираются из соединений, располагаются на отдельной сетке.
          • Для каждого пина компонента генерируется провод (WIRE), соединяющий его с соответствующим флагом.

        Возвращается содержимое .asc файла без дополнительных строк (.backanno, .end).
        """
        # Настройки сетки для компонентов (все значения кратны 16)
        comp_start_x = 48
        comp_start_y = 48
        comp_spacing_x = 144
        comp_spacing_y = 96
        comp_columns = 5

        # Настройки сетки для флагов (узлов)
        flag_start_x = -320
        flag_start_y = 0
        flag_spacing_x = 96
        flag_spacing_y = 48
        flag_columns = 4

        asc_lines = []
        asc_lines.append("Version 4")
        asc_lines.append(sheet_size)

        # Собираем уникальные узлы и список соединений (для проводов)
        node_set = set()
        connection_list = []  # Каждый элемент: (node, inst_name, pin_index, pin_x, pin_y)
        comp_counter = 0

        comp_blocks = []
        text_blocks = []

        for line in net_content.splitlines():
            line = line.strip()
            if not line:
                continue
            # Пропускаем заголовки (например, путь, версия)
            if line.startswith("*") and ("LTspice" in line or ".asc" in line):
                continue
            # Обработка комментариев – преобразуем в текстовый блок
            if line.startswith("*"):
                comment = line[1:].strip()
                text_blocks.append(f"{comment_text_default} ;{comment}")
                continue
            # Обработка директив (.ac, .tran и т.п.) – в текстовый блок
            if line.startswith(".") and not (line.startswith(".backanno") or line.startswith(".end")):
                text_blocks.append(f"{directive_text_coords} !{line}")
                continue

            parts = line.split()
            if len(parts) < 4:
                continue
            inst_name = parts[0]
            # Нормализуем имя для инстанций, начинающихся с "X"
            if inst_name.startswith("X"):
                inst_name = inst_name.replace("§", "")

            mapping = get_default_mapping(inst_name, line)
            pin_count = mapping.get("pin_count", 2)

            # Предполагаем, что первые pin_count токенов после имени – это узлы
            node_list = parts[1:1 + pin_count]
            for node in node_list:
                if node != "0":
                    node_set.add(node)

            # Определяем позицию компонента по сетке
            col = comp_counter % comp_columns
            row = comp_counter // comp_columns
            cx = round16(comp_start_x + col * comp_spacing_x)
            cy = round16(comp_start_y + row * comp_spacing_y)
            comp_counter += 1

            coords = f"{cx} {cy} R0"
            comp_blocks.append(f"SYMBOL {mapping['symbol']} {coords}")
            for win in mapping.get("windows", []):
                comp_blocks.append(win)
            comp_blocks.append(f"SYMATTR InstName {inst_name}")
            comp_value = " ".join(parts[1 + pin_count:])
            comp_blocks.append(f"SYMATTR Value {comp_value}")
            if "extra" in mapping:
                comp_blocks.append(mapping["extra"])

            # Вычисляем позиции пинов компонента и сохраняем соединения
            pin_positions = get_pin_positions((cx, cy), pin_count)
            for i, node in enumerate(node_list):
                if i < len(pin_positions):
                    pin_x, pin_y = pin_positions[i]
                    connection_list.append((node, inst_name, i, pin_x, pin_y))

        # Генерация флагов для каждого уникального узла
        auto_flag_blocks = []
        flag_coord = {}
        for idx, node in enumerate(sorted(node_set)):
            col = idx % flag_columns
            row = idx // flag_columns
            fx = round16(flag_start_x + col * flag_spacing_x)
            fy = round16(flag_start_y + row * flag_spacing_y)
            flag_coord[node] = (fx, fy)
            auto_flag_blocks.append(f"FLAG {fx} {fy} {node}")

        # Генерация проводов, соединяющих пины с флагами
        auto_wire_blocks = []
        for (node, inst_name, pin_index, pin_x, pin_y) in connection_list:
            if node == "0":
                fx, fy = 0, 0
            elif node in flag_coord:
                fx, fy = flag_coord[node]
            else:
                continue
            auto_wire_blocks.append(f"WIRE {int(pin_x)} {int(pin_y)} {int(fx)} {int(fy)}")

        asc_lines.extend(auto_flag_blocks)
        asc_lines.extend(auto_wire_blocks)
        asc_lines.extend(comp_blocks)
        asc_lines.extend(list(map(lambda x: x.replace("\\n", "\\\\n"), text_blocks)))

        return "\n".join(asc_lines)

    def netlist_b64_to_asc_with_description(netlist: str) -> str:
        return f"Here's the .asc file contents:\n\n{net_to_asc(base64.b64decode(netlist.encode("utf-8")).decode("utf-8"))}"

    return Tool(
        name="netlist_to_asc",
        description="Converts base64 representation of netlist to .asc file content.",
        func=lambda query: netlist_b64_to_asc_with_description(query)
    )


def send_asc_to_user_tool(chat_id: int, bot: TeleBot):

    def send_asc_to_user(chat_id: int, bot: TeleBot, asc_file_content: str):
        file_obj = io.BytesIO(asc_file_content.encode("iso-8859-1"))
        file_obj.name = "circuit.asc"
        bot.send_document(chat_id, file_obj)

        return "Successfully sent .asc file contents to the user!"

    return Tool(
        name="send_asc_to_user",
        description="Sends .asc file content to the user.",
        func=lambda asc_file_content: send_asc_to_user(chat_id, bot, asc_file_content)
    )


def send_netlist_b64_to_user_tool(chat_id: int, bot: TeleBot):

    def send_netlist_b64_to_user(chat_id: int, bot: TeleBot, netlist: str):
        print("!!!!!!!")
        print(netlist)
        print("!!!!!!!")

        netlist_str = base64.b64decode(netlist.encode("utf-8")).decode("utf-8")
        file_obj = io.BytesIO(netlist_str.encode("utf-8"))
        file_obj.name = "circuit.net"
        bot.send_document(chat_id, file_obj)

        return "Successfully sent netlist contents to the user!"

    return Tool(
        name="send_netlist_to_user",
        description="Sends base64 representation of netlist as a file to the user.",
        func=lambda asc_file_content: send_netlist_b64_to_user(chat_id, bot, asc_file_content)
    )

