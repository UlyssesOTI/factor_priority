import tkinter as tk
from tkinter import ttk
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
import math
import matplotlib

matplotlib.rcParams['figure.dpi'] = 100


class FactorAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Визначення пріоритетності факторів")
        self.root.geometry("1400x950")
        self.root.configure(bg='#f0f0f0')

        self.factor_count = 0
        self.influences = []
        self.ranking_data = {}
        self.matrix_entries = {}
        self.setup_widgets()

    def setup_widgets(self):
        top_bar = ttk.Frame(self.root, padding="15")
        top_bar.pack(side='top', fill='x')

        ttk.Label(top_bar, text="Позначення факторів:").pack(side='left', padx=5)
        self.prefix_var = tk.StringVar(value="X")
        ttk.Entry(top_bar, textvariable=self.prefix_var, width=5, justify='center').pack(side='left', padx=5)

        ttk.Label(top_bar, text="Кількість факторів:").pack(side='left', padx=15)
        self.factor_count_var = tk.StringVar(value="0")
        ttk.Spinbox(top_bar, from_=0, to=25, textvariable=self.factor_count_var, width=5).pack(side='left', padx=5)

        ttk.Button(top_bar, text="Створити матрицю", command=self.create_matrix).pack(side='left', padx=10)
        ttk.Button(top_bar, text="Визначити ранги", command=self.define_ranks_action).pack(side='left', padx=5)

        self.btn_build_model = ttk.Button(top_bar, text="Побудувати модель пріоритетності",
                                          command=self.build_priority_model, state="disabled")
        self.btn_build_model.pack(side='left', padx=20)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=15, pady=(20, 15))

        self.tab_main = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_main, text=" Матриця та семантична мережа ")

        self.pw1 = ttk.PanedWindow(self.tab_main, orient='horizontal')
        self.pw1.pack(fill='both', expand=True, padx=5, pady=5)

        m_container = ttk.LabelFrame(self.pw1, text="Матриця досяжності", padding=10)
        self.pw1.add(m_container, weight=1)

        self.matrix_canvas = tk.Canvas(m_container, bg='white', highlightthickness=0)
        self.matrix_inner = ttk.Frame(self.matrix_canvas)

        self.matrix_window_id = self.matrix_canvas.create_window((0, 0), window=self.matrix_inner, anchor="nw")

        self.matrix_canvas.pack(fill='both', expand=True)

        self.matrix_inner.bind("<Configure>",
                               lambda e: self.matrix_canvas.configure(scrollregion=self.matrix_canvas.bbox("all")))
        self.matrix_canvas.bind("<Configure>", self._resize_matrix_frame)

        self.s_frame = ttk.LabelFrame(self.pw1, text="Семантична мережа", padding=10)
        self.pw1.add(self.s_frame, weight=3)

        self.root.after(100, self.set_initial_sash_position)

        self.tab_trees = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_trees, text=" Аналіз ієрархії та ранжування ")

        self.rank_table_frame = ttk.LabelFrame(self.tab_trees, text="Ранжування факторів", padding=10)
        self.rank_table_frame.pack(side='top', fill='x', padx=10, pady=5)

        self.rank_canvas_tbl = tk.Canvas(self.rank_table_frame, bg='white', height=210, highlightthickness=0)
        self.rank_table_inner = ttk.Frame(self.rank_canvas_tbl)

        self.rank_window_id = self.rank_canvas_tbl.create_window((0, 0), window=self.rank_table_inner, anchor="nw")

        self.rank_canvas_tbl.pack(fill='both', expand=True)

        self.rank_table_inner.bind("<Configure>", lambda e: self.rank_canvas_tbl.configure(
            scrollregion=self.rank_canvas_tbl.bbox("all")))
        self.rank_canvas_tbl.bind("<Configure>", self._resize_rank_frame)

        analysis_controls = ttk.Frame(self.tab_trees, padding=10)
        analysis_controls.pack(side='top', fill='x')
        ttk.Label(analysis_controls, text="Оберіть фактор:").pack(side='left', padx=5)
        self.selected_factor_var = tk.StringVar()
        self.factor_combobox = ttk.Combobox(analysis_controls, textvariable=self.selected_factor_var, state="readonly",
                                            width=10)
        self.factor_combobox.pack(side='left', padx=5)

        ttk.Button(analysis_controls, text="Побудувати дерева зв'язків", command=self.analyze_trees_action).pack(
            side='left', padx=5)

        self.trees_viz_frame = ttk.LabelFrame(self.tab_trees, text="Графічне представлення ієрархії", padding=5)
        self.trees_viz_frame.pack(side='top', fill='both', expand=True, padx=10, pady=5)

        self.trees_viz_frame.columnconfigure(0, weight=1)
        self.trees_viz_frame.columnconfigure(1, weight=1)

        self.lbl_infl = ttk.Label(self.trees_viz_frame, text="", font=("Arial", 10), foreground="#333333")
        self.lbl_infl.grid(row=0, column=0, pady=(10, 5))
        self.lbl_dep = ttk.Label(self.trees_viz_frame, text="", font=("Arial", 10), foreground="#333333")
        self.lbl_dep.grid(row=0, column=1, pady=(10, 5))

        self.canvas_area = ttk.Frame(self.trees_viz_frame)
        self.canvas_area.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.trees_viz_frame.rowconfigure(1, weight=1)

        self.tab_model = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_model, text=" Модель пріоритетності факторів ")

        self.model_frame = ttk.Frame(self.tab_model)
        self.model_frame.pack(fill='both', expand=True)

    def set_initial_sash_position(self):
        try:
            width = self.root.winfo_width()
            if width > 1:
                self.pw1.sashpos(0, int(width * 0.35))
            else:
                self.root.after(100, self.set_initial_sash_position)
        except:
            pass

    def _resize_rank_frame(self, event):
        if event.width > 1:
            self.rank_canvas_tbl.itemconfig(self.rank_window_id, width=event.width - 4)

    def _resize_matrix_frame(self, event):
        if event is None:
            w = self.matrix_canvas.winfo_width()
        else:
            w = event.width
        if self.factor_count > 0 and w > 1:
            self.matrix_canvas.itemconfig(self.matrix_window_id, width=w)

    def create_matrix(self):
        try:
            self.factor_count = int(self.factor_count_var.get())
            if self.factor_count <= 0: return
            self.influences = np.eye(self.factor_count, dtype=int)
            self.refresh_matrix_ui()
            self.update_combo_list()
            self.show_semantic_network()
            for w in self.canvas_area.winfo_children(): w.destroy()
            for w in self.model_frame.winfo_children(): w.destroy()
            self.btn_build_model.config(state="disabled")
            self.lbl_infl.config(text="");
            self.lbl_dep.config(text="")
            self.notebook.select(self.tab_main)
        except:
            pass

    def refresh_matrix_ui(self):
        prefix = self.prefix_var.get()
        for w in self.matrix_inner.winfo_children(): w.destroy()

        for i in range(self.factor_count + 1):
            self.matrix_inner.grid_columnconfigure(i, weight=1)

        for i in range(self.factor_count + 1):
            for j in range(self.factor_count + 1):
                if i == 0 and j == 0: continue

                txt = f"{prefix}{j if i == 0 else i}" if (i == 0 or j == 0) else (str(self.influences[i - 1][j - 1]))
                bg = '#cccccc' if (i == 0 or j == 0) else 'white'

                if i > 0 and j > 0 and i != j:
                    v = tk.StringVar(value=txt)
                    e = tk.Entry(self.matrix_inner, textvariable=v, width=4, justify='center', bd=1, relief="solid")
                    e.grid(row=i, column=j, padx=0, pady=0, sticky='nsew')
                    e.bind('<Key>', lambda event, r=i - 1, c=j - 1, entry=e: self.on_matrix_input(event, r, c, entry))
                    e.bind("<FocusIn>", lambda event, entry=e: entry.select_range(0, tk.END))
                else:
                    tk.Label(self.matrix_inner, text=txt, width=4, bg=bg, font=('Arial', 9, 'bold'), bd=1,
                             relief="solid").grid(row=i, column=j, padx=0, pady=0, sticky='nsew')

        self.matrix_inner.update_idletasks()
        self.matrix_canvas.config(scrollregion=self.matrix_canvas.bbox("all"))

        self.root.after(50, lambda: self._resize_matrix_frame(None))

    def on_matrix_input(self, event, r, c, entry):
        if event.keysym == 'Tab': return
        if event.char in ('0', '1'):
            entry.delete(0, tk.END)
            entry.insert(0, event.char)
            self.influences[r][c] = int(event.char)
            self.show_semantic_network()
            return "break"
        return "break"

    def update_combo_list(self):
        prefix = self.prefix_var.get()
        vals = [f"{prefix}{i + 1}" for i in range(self.factor_count)]
        self.factor_combobox['values'] = vals
        if vals: self.factor_combobox.set(vals[0])

    def define_ranks_action(self):
        if self.factor_count == 0: return
        self.update_ranking_logic()
        self.btn_build_model.config(state="normal")
        self.notebook.select(self.tab_trees)

    def update_ranking_logic(self):
        data = {}
        for k in range(self.factor_count):
            s1, s2, s3, s4 = 0, 0, 0, 0
            for i in range(self.factor_count):
                if i != k and self.influences[k][i] == 1:
                    s1 += 1
                    for j in range(self.factor_count):
                        if j != k and j != i and self.influences[i][j] == 1: s2 += 1
            for i in range(self.factor_count):
                if i != k and self.influences[i][k] == 1:
                    s3 += 1
                    for j in range(self.factor_count):
                        if j != k and j != i and self.influences[j][i] == 1: s4 += 1
            x1, x2, x3, x4 = s1 * 10, s2 * 5, s3 * -10, s4 * -5
            data[k + 1] = {'s1': s1, 's2': s2, 's3': s3, 's4': s4, 'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4}
        max_abs = max([abs(d['X3'] + d['X4']) for d in data.values()]) if data else 0
        for d in data.values(): d['XF'] = d['X1'] + d['X2'] + d['X3'] + d['X4'] + max_abs
        xf_vals = sorted(list(set([d['XF'] for d in data.values()])))
        for d in data.values(): d['rank'] = xf_vals.index(d['XF']) + 1
        max_rank = max([d['rank'] for d in data.values()]) if data else 0
        for d in data.values(): d['priority'] = max_rank - d['rank'] + 1
        self.ranking_data = data
        self.refresh_ranking_table()

    def refresh_ranking_table(self):
        for w in self.rank_table_inner.winfo_children(): w.destroy()
        prefix = self.prefix_var.get()
        headers = ["Фактор", "s1j", "s2j", "s3j", "s4j", f"{prefix}1j", f"{prefix}2j", f"{prefix}3j", f"{prefix}4j",
                   f"{prefix}Fj", "Ранг", "Пріоритет"]

        for col in range(len(headers)):
            self.rank_table_inner.grid_columnconfigure(col, weight=1)

        for c, h in enumerate(headers):
            tk.Label(self.rank_table_inner, text=h, font=("Arial", 9, "bold"), relief="solid", borderwidth=1, width=10,
                     bg='#e0e0e0').grid(row=0, column=c, sticky="nsew")

        for r, (idx, d) in enumerate(self.ranking_data.items(), 1):
            tk.Label(self.rank_table_inner, text=str(idx), relief="solid", borderwidth=1, width=10).grid(row=r,
                                                                                                         column=0,
                                                                                                         sticky="nsew")
            cols = ['s1', 's2', 's3', 's4', 'X1', 'X2', 'X3', 'X4', 'XF', 'rank', 'priority']
            for c, key in enumerate(cols, 1):
                tk.Label(self.rank_table_inner, text=str(d[key]), relief="solid", borderwidth=1, width=10).grid(row=r,
                                                                                                                column=c,
                                                                                                                sticky="nsew")

        self.rank_table_inner.update_idletasks()
        self.rank_canvas_tbl.config(scrollregion=self.rank_canvas_tbl.bbox("all"))

        self.root.after(10, lambda: self._resize_rank_frame(
            type('Event', (object,), {'width': self.rank_canvas_tbl.winfo_width()})))

    def build_priority_model(self):
        for w in self.model_frame.winfo_children(): w.destroy()
        self.notebook.select(self.tab_model)

        prefix = self.prefix_var.get()

        items = []
        for idx, data in self.ranking_data.items():
            items.append({
                'id': f"{prefix}{idx}",
                'priority': data['priority'],
                'rank': data['rank']
            })

        items.sort(key=lambda x: (x['priority'], x['rank']))
        num_items = len(items)

        fig, ax = plt.subplots()
        fig.patch.set_facecolor('white')
        ax.axis('off')

        SCENE_WIDTH = 12
        block_width = 6
        gap_to_spine = 1.0
        total_content_width = block_width + gap_to_spine

        start_x = (SCENE_WIDTH - total_content_width) / 2
        spine_x = start_x
        block_left_x = start_x + gap_to_spine

        box_height = 0.6
        vertical_gap = 0.4
        row_height = box_height + vertical_gap

        top_y = num_items * row_height + 2.0

        header_color = "#404040"
        block_fill = "#f0f0f0"
        block_edge = "#cccccc"
        line_color = "#404040"

        header_y = top_y - 1.0
        title_box = patches.FancyBboxPatch((block_left_x, header_y - box_height / 2), block_width, box_height,
                                           boxstyle="round,pad=0.1", fc=header_color, ec="none")
        ax.add_patch(title_box)

        ax.text(block_left_x + block_width / 2, header_y, "ЯКІСТЬ", color="white", ha="center", va="center",
                fontsize=10, fontweight="bold")

        arrow_tip_x = block_left_x - 0.05
        ax.annotate('', xy=(arrow_tip_x, header_y), xytext=(spine_x, header_y),
                    arrowprops=dict(arrowstyle='-|>,head_width=0.3,head_length=0.6', color=line_color, lw=2))

        current_y = header_y - row_height
        last_arrow_y = None

        for item in items:
            rect = patches.FancyBboxPatch((block_left_x, current_y - box_height / 2), block_width, box_height,
                                          boxstyle="round,pad=0.1", fc=block_fill, ec=block_edge)
            ax.add_patch(rect)

            label = f"{item['id']} — Пріоритет: {item['priority']} (Ранг: {item['rank']})"
            ax.text(block_left_x + 0.3, current_y, label, color="black", ha="left", va="center", fontsize=9,
                    fontweight="bold")

            arrow_tail_x = block_left_x - 0.05
            ax.annotate('', xy=(spine_x, current_y), xytext=(arrow_tail_x, current_y),
                        arrowprops=dict(arrowstyle='-|>,head_width=0.3,head_length=0.6', color=line_color, lw=2))

            last_arrow_y = current_y
            current_y -= row_height

        if last_arrow_y is not None:
            ax.plot([spine_x, spine_x], [header_y, last_arrow_y], color=line_color, lw=2)

        ax.set_xlim(0, SCENE_WIDTH)
        ax.set_ylim(current_y, top_y)

        canvas = FigureCanvasTkAgg(fig, master=self.model_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        plt.close(fig)

    def analyze_trees_action(self):
        if not self.selected_factor_var.get(): return
        for w in self.canvas_area.winfo_children(): w.destroy()
        sel_val = self.selected_factor_var.get()
        prefix = self.prefix_var.get()
        try:
            root_idx = int(sel_val.replace(prefix, "")) - 1
        except:
            return

        self.lbl_infl.config(text="Граф прямих та непрямих впливів")
        self.lbl_dep.config(text="Граф прямих та непрямих залежностей")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        fig.patch.set_facecolor('white')
        self.draw_relaxed_tree(ax1, root_idx, "influences")
        self.draw_relaxed_tree(ax2, root_idx, "dependencies")
        plt.subplots_adjust(left=0.01, right=0.99, top=1.0, bottom=0.01, wspace=0.1)
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_area)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        plt.close(fig)

    def draw_relaxed_tree(self, ax, root, mode):
        ax.axis('off');
        ax.set_aspect('equal')
        prefix = self.prefix_var.get()

        root_color = '#333333'
        level2_color = '#808080'
        level3_color = '#a6a6a6'
        solid_arrow = '#404040'
        dashed_arrow = '#999999'

        direct = [i for i in range(self.factor_count) if i != root and (
            self.influences[root][i] == 1 if mode == "influences" else self.influences[i][root] == 1)]

        if not direct:
            msg = "Впливи відсутні" if mode == "influences" else "Залежності відсутні"
            ax.text(5, 8.5, msg, ha='center', va='center', fontsize=11, color='#666666', style='italic')
            ax.set_xlim(0, 10);
            ax.set_ylim(5, 12)
            return

        pos = {root: (5, 10)}

        num_d = len(direct)
        for i, d_node in enumerate(direct):
            x_d = 0.5 + (i * 9.0 / (num_d - 1)) if num_d > 1 else 5
            pos[d_node] = (x_d, 7.2)

        children_map = {}
        for d_node in direct:
            children = [t for t in range(self.factor_count) if t != root and t != d_node and (
                self.influences[d_node][t] == 1 if mode == "influences" else self.influences[t][d_node] == 1)]
            for c in children:
                if c not in children_map: children_map[c] = []
                children_map[c].append(pos[d_node][0])

        nodes_l3 = []
        for c, parent_xs in children_map.items():
            ideal = np.mean(parent_xs)
            nodes_l3.append({'id': c, 'x': ideal})

        nodes_l3.sort(key=lambda k: k['x'])
        min_dist = 1.4

        for _ in range(10):
            nodes_l3.sort(key=lambda k: k['x'])
            for i in range(len(nodes_l3) - 1):
                n1 = nodes_l3[i]
                n2 = nodes_l3[i + 1]
                dist = n2['x'] - n1['x']
                if dist < min_dist:
                    overlap = min_dist - dist
                    n1['x'] -= overlap / 2
                    n2['x'] += overlap / 2

        for n in nodes_l3:
            pos[('L3', n['id'])] = (n['x'], 4.4)

        NODE_RADIUS = 0.65

        def draw_arrow(p1, p2, style='solid'):
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = math.hypot(dx, dy)
            if dist == 0: return
            off_x = (dx / dist) * NODE_RADIUS
            off_y = (dy / dist) * NODE_RADIUS
            real_start = (p1[0] + off_x, p1[1] + off_y)
            real_end = (p2[0] - off_x, p2[1] - off_y)
            ax.annotate('', xy=real_end, xytext=real_start,
                        arrowprops=dict(arrowstyle='-|>,head_width=0.2,head_length=0.4',
                                        color=solid_arrow if style == 'solid' else dashed_arrow,
                                        lw=1.5 if style == 'solid' else 1.1,
                                        ls='-' if style == 'solid' else '--'))

        for key, p in pos.items():
            if isinstance(key, tuple):
                n_idx = key[1]; color = level3_color
            elif isinstance(key, int):
                n_idx = key; color = root_color if key == root else level2_color
            else:
                continue

            ax.add_patch(plt.Circle(p, NODE_RADIUS, color=color, zorder=10))
            ax.text(p[0], p[1], f"{prefix}{n_idx + 1}", color='white', ha='center', va='center', fontweight='normal',
                    fontsize=8, zorder=11)

            if key == root:
                for d in direct:
                    if mode == "influences":
                        draw_arrow(p, pos[d], 'solid')
                    else:
                        draw_arrow(pos[d], p, 'solid')

            if isinstance(key, int) and key != root:
                targets = [t for t in range(self.factor_count) if t != root and t != key and (
                    self.influences[key][t] == 1 if mode == "influences" else self.influences[t][key] == 1)]
                for t in targets:
                    if ('L3', t) in pos:
                        if mode == "influences":
                            draw_arrow(p, pos[('L3', t)], 'dashed')
                        else:
                            draw_arrow(pos[('L3', t)], p, 'dashed')

        ax.set_xlim(-5.0, 15.0);
        ax.set_ylim(2.5, 11.5)

    def show_semantic_network(self):
        for w in self.s_frame.winfo_children(): w.destroy()
        if self.factor_count == 0: return
        prefix = self.prefix_var.get()
        fig, ax = plt.subplots(constrained_layout=True)
        G = nx.DiGraph();
        G.add_nodes_from(range(self.factor_count))
        for i in range(self.factor_count):
            for j in range(self.factor_count):
                if i != j and self.influences[i][j] == 1: G.add_edge(i, j)
        pos = nx.circular_layout(G)

        node_fill = '#d9d9d9'
        arrow_color = '#666666'
        label_border = '#cccccc'

        nx.draw_networkx_nodes(G, pos, node_size=800, node_color=node_fill, ax=ax)
        nx.draw_networkx_labels(G, pos, {i: f"{prefix}{i + 1}" for i in range(self.factor_count)}, font_size=9,
                                font_weight='normal', ax=ax)
        for idx, (u, v) in enumerate(G.edges()):
            rad = 0.15 if G.has_edge(v, u) else 0.05
            arrow = patches.FancyArrowPatch(pos[u], pos[v], connectionstyle=f"arc3,rad={rad}", arrowstyle='-|>',
                                            mutation_scale=15, color=arrow_color, shrinkA=22, shrinkB=22)
            ax.add_patch(arrow)
            p1, p2 = np.array(pos[u]), np.array(pos[v])
            mid = p1 + (p2 - p1) * (0.5 + (0.1 if idx % 2 == 0 else -0.1))
            ax.text(mid[0], mid[1], f"{prefix}{u + 1}→{prefix}{v + 1}", fontsize=7, fontweight='normal', ha='center',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=label_border, alpha=0.8))
        ax.axis('off')
        FigureCanvasTkAgg(fig, master=self.s_frame).get_tk_widget().pack(fill='both', expand=True);
        plt.close(fig)


if __name__ == "__main__":
    app_root = tk.Tk()
    FactorAnalysisApp(app_root).root.mainloop()