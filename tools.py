from langchain_community.vectorstores import FAISS
from langchain_core.tools import Tool

from misc import get_default_mapping, round16, get_pin_positions


def simple_circuits_description_to_filenames_tool(vector_store: FAISS):
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

        return f"Here are three filenames and descriptions which fit the best to the description you provided:\n\n{relevant_descriptions_text}Use them to choose filename to get netlist of the circuit."

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


def filename_to_netlist_tool():
    def get_content(query):
        return open(query.removesuffix(".desc.txt") + ".net").read()

    return Tool(
        name="filename_to_netlist",
        description="Returns the netlist of circuit based on the filename of description.",
        func=lambda query: get_content(query)
    )


def netlist_to_asc_tool(sheet_size: str = "SHEET 1 800 600",
                        directive_text_coords: str = "TEXT 450 300 Left 2",
                        comment_text_default: str = "TEXT 80 -250 Left 4"):

    def get_asc_contents(net_content: str):
        # Параметры для сетки компонентов (координаты кратны 16)
        comp_start_x = 48
        comp_start_y = 48
        comp_spacing_x = 144  # кратно 16
        comp_spacing_y = 96  # кратно 16
        comp_columns = 5

        # Параметры для сетки флагов (узлов)
        flag_start_x = -320
        flag_start_y = 0
        flag_spacing_x = 96
        flag_spacing_y = 48
        flag_columns = 4

        asc_lines = []
        asc_lines.append("Version 4")
        asc_lines.append(sheet_size)

        # Собираем уникальные узлы для флагов и список соединений (пин ↔ узел)
        node_set = set()
        connection_list = []  # (node, inst_name, pin_index, pin_x, pin_y)
        comp_positions = {}  # inst_name -> (x, y)

        comp_blocks = []
        text_blocks = []
        comp_counter = 0

        # Обработка строк .net файла
        for line in net_content.splitlines():
            line = line.strip()
            if not line:
                continue
            # Пропускаем заголовочные строки (содержащие путь, версию)
            if line.startswith("*") and ("LTspice" in line or ".asc" in line):
                continue
            # Если строка-комментарий
            if line.startswith("*"):
                comment = line[1:].strip()
                text_blocks.append(f"{comment_text_default} ;{comment}")
                continue
            # Если директива (.ac, .tran и т.п.) (исключая .backanno и .end)
            if line.startswith(".") and not (line.startswith(".backanno") or line.startswith(".end")):
                text_blocks.append(f"{directive_text_coords} !{line}")
                continue

            parts = line.split()
            if len(parts) < 4:
                continue
            inst_name = parts[0]
            # Если имя начинается с "X", удаляем лишние символы, например "§"
            if inst_name.startswith("X"):
                inst_name = inst_name.replace("§", "")

            # Определяем число контактов; по умолчанию 2
            mapping = get_default_mapping(inst_name, line)
            pin_count = mapping.get("pin_count", 2)

            # Предполагаем, что первые pin_count токенов после имени – это узлы
            node_list = parts[1:1 + pin_count]
            for node in node_list:
                if node != "0":
                    node_set.add(node)

            # Генерируем блок компонента
            col = comp_counter % comp_columns
            row = comp_counter // comp_columns
            cx = round16(comp_start_x + col * comp_spacing_x)
            cy = round16(comp_start_y + row * comp_spacing_y)
            comp_positions[inst_name] = (cx, cy)
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

            # Вычисляем позиции пинов компонента
            pin_positions = get_pin_positions((cx, cy), pin_count)
            for i, node in enumerate(node_list):
                if i < len(pin_positions):
                    pin_x, pin_y = pin_positions[i]
                    connection_list.append((node, inst_name, i, pin_x, pin_y))

        # Генерируем флаги (узлы) автоматически
        auto_flag_blocks = []
        flag_coord = {}
        flag_list_generated = sorted(list(node_set))
        flag_counter = 0
        for node in flag_list_generated:
            col = flag_counter % flag_columns
            row = flag_counter // flag_columns
            fx = round16(flag_start_x + col * flag_spacing_x)
            fy = round16(flag_start_y + row * flag_spacing_y)
            flag_coord[node] = (fx, fy)
            auto_flag_blocks.append(f"FLAG {fx} {fy} {node}")
            flag_counter += 1

        # Генерируем провода, соединяющие пины с флагами
        auto_wire_blocks = []
        for (node, inst_name, pin_index, pin_x, pin_y) in connection_list:
            if node == "0":
                fx, fy = (0, 0)
            elif node in flag_coord:
                fx, fy = flag_coord[node]
            else:
                continue
            auto_wire_blocks.append(f"WIRE {int(pin_x)} {int(pin_y)} {int(fx)} {int(fy)}")

        # Собираем итоговый файл: сначала флаги, затем провода, потом компоненты и текстовые блоки
        asc_lines.extend(auto_flag_blocks)
        asc_lines.extend(auto_wire_blocks)
        asc_lines.extend(comp_blocks)
        asc_lines.extend(text_blocks)

        return "\n".join(asc_lines)

    return Tool(
        name="netlist_to_asc",
        description="Returns the .asc file content which is derived from netlist.",
        func=lambda query: get_asc_contents(query)
    )
