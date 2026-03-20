# %% [markdown]
# 
# # 1. 描述性统计与数据分布 (Descriptive Statistics)
# 在展示深奥的模型之前，你必须先证明你了解你的数据。这包括中心趋势、离散程度以及变量间的初步相关性。
# 
# * 展示重点： 均值/中位数、标准差、偏度/峰度、相关性热力图。
# * 常用图表： 直方图 (Histogram)、箱线图 (Boxplot)、热力图 (Heatmap)。(具体见大二上python课件)
# 

# %% [markdown]
# ## 偏度（skewness）
# 
# 定義：描述資料分布的左右對稱程度。
# $$skew(X)=E\left[\left({X-\mu}\over \sigma\right)^3\right]$$
# 值代表：
# * 偏度 > 0 (正偏/右偏)：分布左側集中，右側有長尾巴，平均數 > 中位數。
# * 偏度 < 0 (負偏/左偏)：分布右側集中，左側有長尾巴，平均數 < 中位數。
# * 偏度 = 0：分布左右對稱，接近正態分布。
#   
# 應用：判斷模型假設是否成立，若偏度大可能需數據轉換。（对数中心变换） 
# 
# ![skew](https://img-blog.csdn.net/20170405214815843?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGJtYXRyaXg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

# %% [markdown]
# ## 峰度（peakedness）
# 
# 表征概率密度分布曲线在平均值处峰值高低的特征数。直观看来，峰度反映了峰部的尖度。
# 
# $$Kurt(X)=E\left[\left({X-\mu}\over \sigma \right)^4\right]$$
# 
# 峰度包括正态分布（峰度值=3），厚尾（峰度值>3），瘦尾（峰度值<3）
# 
# ![peakedness](https://img-blog.csdn.net/20170405215819279?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGJtYXRyaXg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

# %% [markdown]
# # 2. 推断性统计与假设检验 (Inferential Statistics)
# 这是统计科研的灵魂。你需要证明你的发现不是由于随机误差造成的。
# 
# * 展示重点： p值 (p-value)、置信区间 (Confidence Intervals)、效应量 (Effect Size)。
# * 常用方法： t检验、ANOVA、卡方检验等。（具体见hypothesis_test）

# %% [markdown]
# # 3.常规绘图包装好的函数

# %%
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import math
# 建立 Science 风格的主题配置
def set_science_style():
    sns.set_theme(
        style="ticks",           # 经典刻度风格
        font="serif",           # 使用衬线字体
        rc={
            "font.serif": ["Times New Roman", "DejaVu Serif"], # 优先使用 Times
            "axes.spines.top": False,    # 去掉上边框
            "axes.spines.right": False,  # 去掉右边框
            "grid.linestyle": "--",      # 网格改用虚线
            "axes.grid": False,          # 默认不显示网格，除非特定需要
            "xtick.direction": "in",     # 刻度线向内
            "ytick.direction": "in",
            "figure.dpi": 800            # 高分辨率输出
        }
    )

set_science_style()

# %%



def plot_stat_regression_ax(ax, x, y, x_label="Independent Variable", y_label="Dependent Variable", title="Linear Regression"):
    """
    针对指定 axes 的科研回归绘图函数
    """
    set_science_style()
    # 2. 统计计算
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value**2
    
    # 排序以便绘制平滑的置信区间阴影
    idx = np.argsort(x)
    x_s, y_s = x[idx], y[idx]
    line = slope * x_s + intercept

    # 3. 绘图：将所有 plt.xxx 替换为 ax.xxx
    ax.scatter(x, y, color="#87C3E4", s=25, label='Data Points')
    ax.plot(x_s, line, color='#F48892', lw=2, label='Fitted Line')

    # 4. 置信区间计算
    n = len(x)
    dof = n - 2
    t_crit = stats.t.ppf(0.95, dof)
    resid = y - (slope * x + intercept)
    s_err = np.sqrt(np.sum(resid**2) / dof)
    
    ci = t_crit * s_err * np.sqrt(1/n + (x_s - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    ax.fill_between(x_s, line - ci, line + ci, color='#e74c3c', alpha=0.15, label='95% CI')

    # 5. 统计标注
    stat_text = f"$R^2 = {r_squared:.3f}$\n$p = {p_value:.4g}$\n$N = {n}$"
    ax.text(0.05, 0.95, stat_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='#bdc3c7'))

    # 6. 细节装饰
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # ax.grid(True, linestyle=':', alpha=0.6)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    ax.legend()

# --- 如何调用（组合多个子图） ---
if __name__ == "__main__":
    # 创建 1行2列 的布局
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=850)
    set_science_style()
    # 模拟数据 A
    x1 = np.random.rand(40) * 10
    y1 = 2 * x1 + 5 + np.random.randn(40) * 2
    plot_stat_regression_ax(ax1, x1, y1, title="Group A: Strong Correlation")

    # 模拟数据 B
    x2 = np.random.rand(40) * 10
    y2 = 0.5 * x2 + 10 + np.random.randn(40) * 5
    plot_stat_regression_ax(ax2, x2, y2, title="Group B: Weak Correlation")

    plt.tight_layout()
    plt.show()

# %%

def plot_pure_plt_joint_subfig(subfig, x, y, x_label="Independent Variable", y_label="Dependent Variable"):
    """
    在指定的子画布 (subfig) 上绘制 Jointplot
    """
    # 1. 在子画布内部定义网格
    gs = GridSpec(2, 2, figure=subfig,
                  width_ratios=[4, 1], height_ratios=[1, 4], 
                  hspace=0.08, wspace=0.08)
    
    # 2. 创建子图
    ax_main = subfig.add_subplot(gs[1, 0])
    ax_hist_x = subfig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_hist_y = subfig.add_subplot(gs[1, 1], sharey=ax_main)

    
    # --- 绘制主图 (ax_main) ---
    ax_main.scatter(x, y, color="#87C3E4", edgecolor="#6EB8E0",s=30)
    
    # 线性回归计算
    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    x_range = np.linspace(x.min(), x.max(), 100)
    ax_main.plot(x_range, slope * x_range + intercept, color='#F48892', lw=2)
    
    # 统计信息标注
    stats_text = f'$R^2 = {r_value**2:.2f}$\n$p = {p_value:.2g}$'
    ax_main.text(0.05, 0.95, stats_text, transform=ax_main.transAxes, 
                 verticalalignment='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='#bdc3c7'))

    # --- 绘制边缘分布 (直方图) ---
    ax_hist_x.hist(x, bins=20, color="#87C3E4", edgecolor='white')
    ax_hist_x.axis('off')

    ax_hist_y.hist(y, bins=20, color="#87C3E4", edgecolor='white', orientation='horizontal')
    ax_hist_y.axis('off')

    # --- 细节微调 ---
    ax_main.set_xlabel(x_label)
    ax_main.set_ylabel(y_label)
    ax_main.grid(True, linestyle=':', alpha=0.6)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)

# --- 如何在多图布局中使用 ---
if __name__ == "__main__":
    # 创建一个包含两列的大图
    fig = plt.figure(figsize=(14, 6), dpi=800)
    
    # 划分为左右两个子画布 (Subfigures)
    subfigs = fig.subfigures(1, 2, wspace=0.1)
    
    # 数据 A
    x1 = np.random.normal(50, 10, 200)
    y1 = 0.8 * x1 + np.random.normal(0, 5, 200)
    plot_pure_plt_joint_subfig(subfigs[0], x1, y1, x_label="Control Group X", y_label="Outcome Y")
    
    # 数据 B
    x2 = np.random.normal(40, 15, 200)
    y2 = -0.4 * x2 + 100 + np.random.normal(0, 10, 200)
    plot_pure_plt_joint_subfig(subfigs[1], x2, y2, x_label="Treatment Group X", y_label="Outcome Y")

    plt.suptitle("Comparative Joint Distribution Analysis", fontsize=16)
    # 建议保存为 PDF 或 EPS，这是主流期刊最喜欢的格式
    plt.savefig("my_research_plot.pdf", 
            format='pdf', 
            dpi=600,              # 虽然矢量图不依赖像素，但内部的位图元素会按此分辨率渲染
            bbox_inches='tight',  # 关键：自动裁剪多余的白边
            transparent=True)     # 背景透明，方便排版
    plt.show()

# %%


def plot_scientific_pairplot(df, vars_list, hue_col=None, save_name="pairplot_result"):
    """
    科研专用多变量矩阵图封装函数
    
    参数:
    df: pd.DataFrame, 数据集
    vars_list: list, 需要分析的连续变量列名
    hue_col: str, 分组变量列名 (如 'Group', 'Status')
    save_name: str, 保存的文件名前缀
    """
    
    # 1. 全局科研风格配置 (Science/Nature 风格)
    set_science_style()

    # 2. 绘制 pairplot
    # kind='reg': 自动绘制回归线和置信区间
    # diag_kind='kde': 对角线显示平滑的核密度估计
    # corner=True: 【关键】只显示下三角矩阵，减少冗余，科研论文首选
    g = sns.pairplot(
        df, 
        vars=vars_list, 
        hue=hue_col, 
        kind='reg', 
        diag_kind='kde',
        palette=["#91CAE8",'#F48892'], 
        corner=False,  
        plot_kws={
            'scatter_kws': {'alpha': 0.7, 's': 20, 'rasterized': True}, # 散点位图化减少PDF体积
            'line_kws': {'lw': 1.5}
        },
        diag_kws={'fill': True, 'alpha': 0.7}
    )

    # 3. 细节修饰
    # 移除多余边框
    sns.despine(fig=g.fig, trim=False)
    
    # 调整标题 (由于 corner=True，标题位置需要微调)
    g.fig.suptitle(f"Multivariate Correlation Analysis: {', '.join(vars_list)}", 
                   y=1.02, fontsize=14, fontweight='bold')

    # 4. 自动保存矢量图
    # 保存为 PDF 供投稿，保存为 PNG 供快速查看
    try:
        g.savefig(f"{save_name}.pdf", bbox_inches='tight', dpi=800)
        print(f"✅ 图表已成功保存为: {save_name}.pdf")
    except Exception as e:
        print(f"❌ 保存失败: {e}")

    plt.show()

# --- 使用示例 ---
if __name__ == "__main__":
    # 1. 模拟科研数据
    import numpy as np
    np.random.seed(42)
    n = 150
    data = pd.DataFrame({
        'Feature_A': np.random.normal(10, 2, n),
        'Feature_B': np.random.normal(20, 5, n),
        'Feature_C': np.random.normal(30, 8, n),
        'Group': np.random.choice(['Control', 'Treatment'], n)
    })
    # 制造变量间相关性
    data['Feature_B'] += 0.6 * data['Feature_A']
    data['Feature_C'] -= 0.4 * data['Feature_B']

    # 2. 调用封装函数
    plot_scientific_pairplot(
        df=data, 
        vars_list=['Feature_A', 'Feature_B', 'Feature_C'], 
        hue_col='Group',
        save_name="Scientific_Pairplot_Output"
    )

# %%


def plot_custom_heatmap(df, title="Correlation Matrix"):
    """
    科研专用自定义颜色热力图
    colors: list, 颜色代码列表。例如 ['#3498db', '#ffffff', '#e74c3c'] (蓝-白-红)
    """
    # 1. 设置科研基础风格
    set_science_style()
    
    # 2. 创建自定义色图 (Colormap)
    colors = [
    "#104680",  # R:016 G:070 B:128
    "#317CB7",  # R:049 G:124 B:183
    "#6DADE5",  # R:109 G:173 B:209
    "#B6D7E8",  # R:182 G:215 B:232
    "#E9F0F4",  # R:233 G:241 B:244
    "#F1E3CE",  # R:251 G:227 B:213
    "#F6B293",  # R:246 G:178 B:147
    "#DC6D57",  # R:220 G:109 B:087
    "#B72230",  # R:183 G:034 B:048
    "#6D011F"   # R:109 G:001 B:031
    ]
    
    # 创建渐变对象，n_bins 为平滑度
    my_cmap = LinearSegmentedColormap.from_list("custom_map", colors, N=20)

    # 3. 计算相关系数矩阵
    corr = df.corr()

    # 4. 绘图
    plt.figure(figsize=(10, 8),dpi=800)
    
    # mask: 只显示下三角矩阵（科研常用，避免视觉干扰）

    sns.heatmap(corr,               
                cmap=my_cmap,           # 自定义颜色
                vmax=1, vmin=-1,        # 确保颜色轴对称
                center=0,               # 中心值
                annot=True,             # 显示数字
                fmt=".2f",              # 数字格式
                linewidths=0.5,         # 格子间距
                cbar_kws={"shrink": .8, "label": "Pearson Correlation (r)"}, # 侧边条配置
                square=True,alpha=0.85)            # 每个格子为正方形

    plt.title(title, fontsize=15, pad=20)
    
    # 5. 保存矢量图
    plt.savefig(f"{title}_heatmap.pdf", bbox_inches='tight')
    plt.show()



def generate_complex_corr_data(n_samples=100):
    np.random.seed(88)
    
    # 1. 生成三个独立的种子变量 (Latent Factors)
    f1 = np.linspace(0, 10, n_samples)
    f2 = np.linspace(10, 0, n_samples)
    f3 = np.random.normal(5, 2, n_samples)
    
    data = {}
    
    # --- 第一组：基于 f1 的强正相关 ---
    data['Var_A'] = f1 + np.random.normal(0, 0.5, n_samples)
    data['Var_B'] = 0.9 * f1 + np.random.normal(0, 0.3, n_samples)
    data['Var_C'] = 1.2 * f1 + np.random.normal(0, 0.8, n_samples)
    
    # --- 第二组：基于 f2 的强正相关 (但与第一组强负相关) ---
    data['Var_D'] = f2 + np.random.normal(0, 0.4, n_samples)
    data['Var_E'] = 0.85 * f2 + np.random.normal(0, 0.2, n_samples)
    
    # --- 第三组：互相关与混合 ---
    # Var_F 负相关于 A，Var_G 正相关于 D
    data['Var_F'] = -0.95 * data['Var_A'] + np.random.normal(0, 0.5, n_samples)
    data['Var_G'] = 0.7 * data['Var_D'] + 0.3 * f3 + np.random.normal(0, 0.5, n_samples)
    
    # --- 第四组：几乎共线性 ---
    data['Var_H'] = data['Var_B'] * 0.98 + np.random.normal(0, 0.05, n_samples)
    
    # --- 第五组：中等相关与独立变量 ---
    data['Var_I'] = 0.5 * f1 + 0.5 * f3 + np.random.normal(0, 1.0, n_samples)
    data['Var_J'] = np.random.normal(10, 2, n_samples) # 相对独立的变量
    
    return pd.DataFrame(data)

# 生成数据
df_complex = generate_complex_corr_data()
plot_custom_heatmap(df_complex,title="heatmap")

# %%

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

four_color_li=["#8EC8ED", "#AED594", "#D693BE", "#F5B3A5"]


def plot_scatter_with_ellipse(x, y, ax=None, n_std=2.0, color="#8EC8ED", label=None, **kwargs):
    """
    封装函数：一键绘制散点图并覆盖带填充的置信椭圆
    
    参数:
    x, y : 数据点
    ax   : matplotlib axes 对象
    n_std: 标准差倍数 (2.0 约等于 95% 置信区间)
    color: 自定义颜色 (Hex 或名称)
    label: 标签，用于显示图例
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    set_science_style()
    # 1. 绘制散点 (设置稍微淡一点，突出椭圆)
    ax.scatter(x, y, s=30, color=color, edgecolors='white', linewidth=0.5)

    # 2. 计算椭圆参数
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    
    # 3. 创建椭圆对象
    # facecolor: 填充色 (alpha 设薄一点)
    # edgecolor: 边缘色 (设为同色系深色或直接同色)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=color, edgecolor=color, alpha=0.5, 
                      linestyle='--', linewidth=1.5, label=label, **kwargs)

    # 4. 矩阵变换（旋转、缩放、平移）
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

    # 5. 自动调整坐标轴范围
    ax.relim()
    ax.autoscale_view()
    
    return ax

# --- 实际调用演示 ---

# 1. 模拟两组不同的数据
np.random.seed(42)
# 第一组：正相关
x1 = np.random.rand(50) * 10
y1 = 1.5 * x1 + 2 + np.random.randn(50) * 2
# 第二组：负相关
x2 = np.random.rand(50) * 10
y2 = -1.2 * x2 + 15 + np.random.randn(50) * 2

# 2. 创建画布
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

# 3. 分别调用函数，传入自定义颜色
plot_scatter_with_ellipse(x1, y1, ax=ax, color="#8EC8ED", label="Group A (High)")
plot_scatter_with_ellipse(x2, y2, ax=ax, color="#D693BE", label="Group B (Low)")

# 4. 修饰
ax.set_title("Scientific Analysis: Scatter with 95% Confidence Ellipses", fontsize=14, pad=15)
ax.set_xlabel("Feature X (units)", fontsize=12)
ax.set_ylabel("Feature Y (units)", fontsize=12)
ax.legend() 

plt.tight_layout()
plt.show()

# %%

def plot_custom_boxplot_plt(ax, data_list, labels, title=""):
    """
    使用纯 matplotlib 绘制带填充色的分类箱线图
    
    参数:
    ax        : 子图对象
    data_list : 包含不同类别数据的列表, 例如 [group1_data, group2_data, ...]
    labels    : 类别名称列表
    """
    set_science_style()
    
    colors=['#257D8B', '#68BED9', '#BFDFD2', '#EAA558', '#ED8D5A', '#EFCE87']

    # 1. 绘制基础箱线图
    # patch_artist=True 必须设置，否则无法填充颜色
    # medianprops 设置中位数线条颜色
    bplot = ax.boxplot(data_list, 
                       patch_artist=True, 
                       labels=labels,
                       showmeans=True,
                       widths=0.5,
                       medianprops={'color': 'black', 'linewidth': 1.5},      #调中位线的样式
                       meanprops={'marker':'+','markeredgecolor':"black", 'markersize':5}, #均值
                       whiskerprops={'color': 'black', 'linewidth': 1.5},  #长出来那条线
                       capprops={'color': 'black', 'linewidth': 1.5},    #长出来横的那条杠
                       boxprops={'color': 'red', 'linewidth': 1.5})   #箱子的边框

    # 2. 为每个箱体循环上色
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)      # 设置透明度，更有质感
        patch.set_edgecolor('black') # 箱体边框色
        patch.set_linewidth(1.2)

    # 3. 细节美化
    ax.set_title(title, fontsize=14, pad=15)
    
    # # 移除顶部和右侧脊柱，显得清爽
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    
    # # 添加虚线水平网格
    # ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)

    return bplot

# --- 准备数据 ---
np.random.seed(10)
g1 = np.random.normal(100, 10, 200)
g2 = np.random.normal(90, 20, 200)
g3 = np.random.normal(110, 15, 200)
g4 = np.random.normal(105, 5, 200)
g5 =np.random.normal(115, 5, 200)
g6=np.random.normal(95, 5, 200)

all_data = [g1, g2, g3, g4,g5,g6]
all_labels = ['Control', 'Group A', 'Group B', 'Group C',"Group D","Group E"]


# --- 绘图 ---
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
plot_custom_boxplot_plt(ax, all_data, all_labels, title="Experiment Results")

plt.ylabel("Value (Units)")
plt.tight_layout()
plt.show()

# %%

six_color_li=['#257D8B', '#68BED9', '#BFDFD2', '#EAA558', '#ED8D5A', '#EFCE87']

# %%

def plot_top_journal_bar(x_pos, ax, data, color="#5DADE2"):
    """
    绘制顶刊风格的分类柱状图组件
    """
    # 处理 DataFrame 或 Series 输入
    if isinstance(data, (pd.Series, pd.DataFrame)):
        data_array = data.values.flatten()
    else:
        data_array = np.array(data)
        
    mean = np.mean(data_array)
    sem = np.std(data_array, ddof=1) / np.sqrt(len(data_array))
    
    # 1. 绘制柱子 (注意：这里不再直接传 label)
    ax.bar(x_pos, mean, color=color, alpha=0.5, width=0.6, 
           edgecolor=color, linewidth=2, zorder=1)
    
    # 2. 绘制误差棒 (yerr 是误差范围)
    ax.errorbar(x_pos, mean, yerr=sem, fmt='none', ecolor='black', 
                capsize=5, elinewidth=1.2, zorder=4)
    
    # 3. 绘制抖动散点 (围绕 x_pos 抖动)
    x_jitter = x_pos + np.random.normal(0, 0.04, size=len(data_array))
    ax.scatter(x_jitter, data_array, color=color, alpha=0.9, s=30, 
               edgecolor='white', linewidth=0.5, zorder=3)

    # 4. 基础修饰
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return ax

# --- 模拟多组数据绘制 ---
np.random.seed(42)
group_a = np.random.normal(15, 3, 20)
group_b = np.random.normal(20, 4, 20)
group_c = np.random.normal(12, 2, 20)

fig, ax = plt.subplots(figsize=(7, 6))

# 手动指定 x 轴位置和颜色
plot_top_journal_bar(0, ax, group_a, color="#23BAC5")
plot_top_journal_bar(1, ax, group_b, color="#EECA40")
plot_top_journal_bar(2, ax, group_c, color="#FD763F")

# --- 关键：统一设置 X 轴标签 ---
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Control', 'Treat A', 'Treat B'], fontsize=12, fontweight='bold')

ax.set_ylabel("Measured Value (Units)", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
color_list = [
"#DB3124",
"#FFDF92",
"#90BEE0",
"#4B74B2"
]
def plot_grouped_journal_bar(df, category_col, value_cols, ax=None, colors=None):
    """
    绘制并列柱状图（均值 + SEM）
    df: 包含数据的DataFrame
    category_col: X轴的分类标签列（如 'Time'）
    value_cols: 需要并列对比的数值列名列表（如 ['Control', 'Treat']）
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # 1. 基础设置
    n_groups = len(df)                 # 共有多少个大类（X轴刻度数）
    n_bars = len(value_cols)           # 每个大类里有多少个并列的柱子
    bar_width = 0.35                   # 柱子宽度
    index = np.arange(n_groups)        # 基准位置：[0, 1, 2...]
    
    if colors is None:
        colors = ["#23BAC5","#EECA40", "#FD763F"] # 经典配色

    # 2. 循环绘制每一组并列柱子
    for i, col in enumerate(value_cols):
        # 计算偏移量：让柱子以 index 为中心对称分布
        # 偏移公式：index + (当前序号 - 总数/2 + 0.5) * 宽度
        offset = (i - (n_bars - 1) / 2) * bar_width
        
        means = df[col]
        # 假设误差是按 SEM 计算（这里用模拟数据，实际可用 df.sem()）
        errors = df[col] * 0.1 
        
        # 绘制柱子
        ax.bar(index + offset, means, bar_width, 
               label=col, color=colors[i], alpha=0.6, 
               edgecolor=colors[i], linewidth=2)
        
        

    # 3. 美化
    ax.set_xticks(index)
    ax.set_xticklabels(df[category_col], fontsize=12, fontweight='bold')
    ax.legend(frameon=True, fontsize=10)
    
    # 移除边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return ax

# --- 模拟数据 ---
data = pd.DataFrame({
    'Time': ['Day 1', 'Day 3', 'Day 7'],
    'Control': [10, 15, 12],
    'Treatment': [12, 22, 30]
})

fig, ax = plt.subplots(figsize=(7, 5))
plot_grouped_journal_bar(data, 'Time', ['Control', 'Treatment'], ax=ax)
plt.ylabel("Intensity (a.u.)", fontsize=12, fontweight='bold')
plt.show()

# %%

def plot_stacked_journal_bar(df, x_col, stack_cols, ax=None, colors=None):
    """
    绘制顶刊风格的堆积柱状图
    stack_cols: 需要堆叠的列名列表，顺序自下而上
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    
    if colors is None:
        colors = ["#DB3124","#FFDF92","#90BEE0","#4B74B2"] 

    # 初始化底部高度为 0
    bottom_height = np.zeros(len(df))
    
    x_pos = np.arange(len(df))
    bar_width = 0.6

    # 循环堆叠
    for i, col in enumerate(stack_cols):
        values = df[col].values
        ax.bar(x_pos, values, bar_width, 
               bottom=bottom_height, 
               label=col, 
               color=colors[i], 
               edgecolor='white', # 用白色边框做分割，更有高级感
               linewidth=1,
               alpha=0.9)
        
        # 更新下一层的底部高度
        bottom_height += values

    # 细节美化
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df[x_col], fontsize=12, fontweight='bold')
    
    # 图例放在右侧或上方
    ax.legend(frameon=True, bbox_to_anchor=(1, 1))
    
    # 移除顶部和右侧线条
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return ax

# --- 模拟数据 (例如：不同细胞类型的占比) ---
data = pd.DataFrame({
    'Sample': ['Site A', 'Site B', 'Site C'],
    'Type_1': [40, 30, 20],
    'Type_2': [35, 40, 30],
    'Type_3': [25, 30, 25]
})

fig, ax = plt.subplots(figsize=(6, 6))
plot_stacked_journal_bar(data, 'Sample', ['Type_1', 'Type_2', 'Type_3'], ax=ax)

plt.ylabel("Relative Abundance (%)", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
#双y轴（y轴尺度不一样），用来对比不同数据的变化趋势
df = pd.read_csv("双Y轴折线图.csv")  # 从指定路径读取CSV文件到DataFrame中

x = df['date']               # 从DataFrame中提取'date'列作为x轴数据
y1 = df['psavert']          # 从DataFrame中提取'psavertt'列作为第一个y轴的数据
y2 = df['unemploy']          # 从DataFrame中提取'unemploy'列作为第二个y轴的数据

# 绘制线条 1（左Y轴）
fig, ax1 = plt.subplots(1, 1, figsize=(14, 6), dpi=300)  # 创建一个大小为14x6英寸，分辨率为300的子图
ax1.plot(x, y1, color='tab:red')  # 在ax1上绘制线条，颜色为红色

# 绘制线条 2（右Y轴）
ax2 = ax1.twinx()  # 实例化一个共享相同X轴的第2个坐标轴
ax2.plot(x, y2, color='tab:blue')  # 在ax2上绘制线条，颜色为蓝色

# 图表修饰
# ax1（左Y轴）
ax1.set_xlabel('Year', fontsize=20)  # 设置x轴标签为'Year'，字体大小为20
ax1.tick_params(axis='x', rotation=0, labelsize=12)  # 设置x轴刻度参数，不旋转，字体大小为12
ax1.set_ylabel('Personal Savings Rate', color='tab:red', fontsize=20)  # 设置y轴标签为'Personal Savings Rate'，颜色为红色，字体大小为20
ax1.tick_params(axis='y', rotation=0, labelcolor='tab:red')  # 设置y轴刻度参数，不旋转，标签颜色为红色
ax1.grid(alpha=.4)  # 显示网格，透明度为0.4

# ax2（右Y轴）
ax2.set_ylabel("# Unemployed (1000's)", color='tab:blue', fontsize=20)  # 设置y轴标签为"# Unemployed (1000's)"，颜色为蓝色，字体大小为20
ax2.tick_params(axis='y', labelcolor='tab:blue')  # 设置y轴刻度参数，标签颜色为蓝色
ax2.set_xticks(np.arange(0, len(x), 60))  # 设置x轴刻度为从0开始，每隔60个单位一个刻度
ax2.set_xticklabels(x[::60], rotation=90, fontdict={'fontsize': 10})  # 设置x轴刻度标签，每隔60个标签一个，旋转90度，字体大小为10
ax2.spines['top'].set_visible(True)
ax2.spines['right'].set_visible(True)
ax2.set_title("Personal Savings Rate vs Unemployed", fontsize=22)  # 设置图表标题为"Personal Savings Rate vs Unemployed"，字体大小为22
fig.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

plt.show()  # 显示图表

# %%
#山脊图 用来比较不同分布
import joypy
from matplotlib import cm




# 1. 生成或加载你的数据 (这里用模拟数据)
np.random.seed(42)
data = pd.DataFrame({
    'Year': np.repeat(range(2010, 2021), 100),
    'Value': np.random.randn(1100).cumsum() + np.tile(np.linspace(0, 10, 11), 100)
})

# 2. 绘制山脊图
fig, axes = joypy.joyplot(
    data, 
    by="Year", 
    column="Value", 
    colormap=cm.autumn,     # 渐变色系 (例如: viridis, plasma, ocean)具体见https://matplotlib.org/stable/users/explain/colors/colormaps.html
    fade=True,              # 是否开启渐变褪色
    alpha=0.75,            # 透明度
    background="#ffffff",    # 背景色
    overlap=2,              # 山脉重叠程度 (越高越紧凑)
    title="this is title",
    figsize=(10, 7)
)
# 关掉上方刻度线 (Ticks) 以及刻度标签 (Labels)
# plt.tick_params(top=False, labeltop=False)
# 3. 细节微调
plt.xlabel("Value Axis", fontsize=12)
plt.show()

# %%
#桑基图见网站，自己可视化制作更简单https://sankeymatic.com/build/
#桑基图用于定性分析流向

# %%
#甜甜圈饼图

def plot_sci_donut(data, labels, title="Composition", save_path=None, colors=None):
    """
    绘制符合 SCI 发表标准的甜甜圈饼图
    
    参数:
    - data: 数值列表
    - labels: 标签列表
    - title: 中间显示的标题
    - save_path: 保存路径 (如 'fig1.pdf')
    - colors: 自定义颜色列表 (可选)
    """
    
    # 1. 默认 SCI 高级配色 (Nature/Science 风格)
    if colors is None:
        colors = ['#2878B5', '#9AC9E0', '#C82423', '#F8AC8C', '#72BA68', '#BA8FBB']
    
    # 2. 环境设置：设置全局字体为无衬线字体（科研论文常用）
    set_science_style()
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    
    # 3. 绘制圆环
    # width=0.35 是 SCI 论文中比例较舒适的厚度
    wedges, texts, autotexts = ax.pie(
        data, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=90, 
        colors=colors,
        pctdistance=0.82, 
        textprops={'fontsize': 10, 'fontweight': 'bold'},
        wedgeprops={'width': 0.35, 'edgecolor': 'w', 'linewidth': 2}
    )
    
    # 4. 精细化调整百分比文字颜色
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(9)

    # 5. 添加中心注释
    ax.text(0, 0, title, ha='center', va='center', 
            fontsize=12, fontweight='bold', color='#333333')

    # 6. 设置图例（SCI 风格通常建议侧边放置图例以保持画面整洁）
    ax.legend(wedges, labels,
              title="Categories",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1),
              frameon=False) # 去掉边框更高级

    ax.axis('equal')  
    
        
    plt.tight_layout()
    plt.show()

# --- 调用示例 ---
my_data = [40, 30, 20, 10]
my_labels = ['Control', 'Treatment A', 'Treatment B', 'Placebo']

plot_sci_donut(my_data, my_labels, title="Cell\nViability")

# %%

from math import pi


def plot_ring_radar(data,criteria_label_list,data_color_list,ring_color_list=['#ADD8E6', '#B0C4DE', '#F08080', '#FFCC99', '#FFE4B5', '#D3EDD3']):

    # 1. 设置中文显示与字体（SCI通常建议Arial）
    plt.rcParams['font.family'] = 'Times New Roman'

    # 2. 准备数据
    labels = criteria_label_list
    num_vars = len(labels)

    # 各模型数据 (示例数据，请根据实际修改)
    # data = {
    #     'RF': [0.75, 0.82, 0.78, 0.80, 0.65, 0.72],
    #     'SVM': [0.65, 0.85, 0.60, 0.75, 0.55, 0.88],
    #     'XGB': [0.88, 0.70, 0.75, 0.72, 0.68, 0.75],
    #     'LGBM': [0.78, 0.75, 0.82, 0.78, 0.62, 0.70]
    # }

    # 闭合数据点
    angles = np.linspace(0, 2 * pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # 3. 绘制雷达图主体
    colors = data_color_list
    for i, (model_name, values) in enumerate(data.items()):
        values += values[:1]
        ax.plot(angles, values, color=colors[i], linewidth=2, label=model_name, marker='o', markersize=4)
        ax.fill(angles, values, color=colors[i], alpha=0.1)

    # 4. 优化坐标轴与背景
    ax.set_theta_offset(pi / 2) # 设置顶点开始位置
    ax.set_theta_direction(-1)  # 顺时针排列
    ax.set_rlabel_position(0)   # 刻度数字的位置

    # 设置网格
    plt.xticks(angles[:-1], []) # 隐藏内部默认标签，我们手动画外部环形
    # plt.yticks([0.28, 0.55, 0.83], ["0.28", "0.55", "0.83"], color="black", size=10)
    # plt.ylim(0, 1)

    # 5. 模拟外部环形彩色标签 (这是原图的精华)
    # 使用不同颜色的弧线模拟外部色块
    bar_height=0.15 #环厚度
    ring_colors = ring_color_list
    for i, (label, color) in enumerate(zip(labels, ring_colors)):
        # angle_start = angles[i] - (pi/num_vars)
        # angle_end = angles[i] + (pi/num_vars)
        
        # 绘制外部色块弧线
        ax.bar(angles[i], bar_height, width=2*pi/num_vars, bottom=1.02, color=color, alpha=0.6, edgecolor='none')
        # 添加文字
        ax.text(angles[i], 1.1, label, ha='center', va='center', rotation=0, weight='bold')

    # 6. 图例设置
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)

    plt.tight_layout()
    ax.set_ylim(0, 1.2)
    plt.show()

# %%

from scipy.special import expit

def plot_smooth_stacked_area(df, colors=None, transition_ratio=0.3, figsize=(11, 6)):
    """
    将 DataFrame 转换为平滑阶梯过渡的百分比堆叠面积图。
    
    参数:
    - df: DataFrame, 第一列应为类别(时间/月份), 其余列为各层数值(总和应为100)
    - colors: 列表, 自定义颜色代码
    - transition_ratio: 过渡带占比 (0到1之间), 越小越接近阶梯图, 越大越圆润
    - figsize: 元组, 画布大小
    """
    
    # 1. 提取数据
    categories = df.iloc[:, 0].values
    data_values = df.iloc[:, 1:].values.T  # 转置以便按行处理各层
    num_layers = data_values.shape[0]
    num_steps = data_values.shape[1]
    
    # 2. 插值逻辑
    points_per_step = 100
    trans_width = int(points_per_step * transition_ratio)
    flat_width = points_per_step - trans_width
    
    y_fine = [[] for _ in range(num_layers)]
    
    for i in range(num_steps - 1):
        for j in range(num_layers):
            # 当前值和下一个值
            y_start, y_end = data_values[j, i], data_values[j, i+1]
            
            # 平直段
            y_fine[j].extend(np.full(flat_width, y_start))
            
            # Sigmoid 过渡段
            x_trans = np.linspace(-10, 10, trans_width)
            y_trans = y_start + (y_end - y_start) * expit(x_trans)
            y_fine[j].extend(y_trans)

    # 补全最后一个分类的平直段
    for j in range(num_layers):
        y_fine[j].extend(np.full(points_per_step, data_values[j, -1]))

    x_fine = np.arange(len(y_fine[0]))
    y_stack = np.array(y_fine)

    # 3. 绘图
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    if colors is None:
        colors = ["#326195", "#A1CADF", "#FEF9C3", "#F2A561", "#C52B28"] # 默认还原配色
        
    ax.stackplot(x_fine, y_stack, 
                 labels=df.columns[1:], 
                 colors=colors[:num_layers], 
                 alpha=0.9, 
                 edgecolor='white', 
                 linewidth=0.3)

    # 4. 样式美化
    # 刻度位置设置在每个平直段和过渡段组成的周期中间
    mid_points = np.arange(0, num_steps * points_per_step, points_per_step) + (points_per_step / 2)
    ax.set_xticks(mid_points)
    ax.set_xticklabels(categories)
    
    ax.set_ylim(0, 100)
    ax.set_xlim(0, x_fine[-1])
    ax.grid(axis='both', color='gray', linestyle='-', alpha=0.1)
    ax.set_axisbelow(True)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.tight_layout()
    
    return fig, ax

# --- 使用示例 ---

# 模拟你的数据源
raw_data = {
    'Month': ["January", "February", "March", "April", "May", "June"],
    'Shade 5': [39, 5, 31, 31, 32, 10],
    'Shade 4': [8, 35, 14, 18, 28, 18],
    'Shade 3': [15, 12, 9, 7, 20, 32],
    'Shade 2': [23, 30, 13, 27, 12, 25],
    'Shade 1': [15, 18, 33, 17, 8, 15] # 确保总和为100
}
df_test = pd.DataFrame(raw_data)

# 调用函数
fig, ax = plot_smooth_stacked_area(df_test, transition_ratio=0.3)
plt.show()

# %%

import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

def plot_q_heatmap(df, cmap_color="#FF5A5F", corner_radius=0.3, cell_gap=0.1, label_size=10):
    """
    绘制带有色条标注的Q版圆角热力图
    """
    rows, cols = df.shape
    # 根据行列数自动调整画布比例
    fig, ax = plt.subplots(figsize=(cols * 0.8 + 2, rows * 0.5 + 1)) 

    # 1. 定义颜色映射
    # 模拟图中从 Low(蓝) 到 Median(白) 再到 High(红) 的过渡
    colors = ["#79C7FF", "#F0F0F0", cmap_color] 
    my_cmap = LinearSegmentedColormap.from_list("q_heatmap", colors)
    
    # 数据归一化
    vmin, vmax = df.min().min(), df.max().max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # 2. 遍历单元格绘制圆角矩形
    for r in range(rows):
        for c in range(cols):
            val = df.iloc[r, c]
            color = my_cmap(norm(val))
            
            # 创建圆角矩形单元格
            rect = patches.FancyBboxPatch(
                (c + cell_gap/2, rows - r - 1 + cell_gap/2), 
                1 - cell_gap, 1 - cell_gap,
                boxstyle=f"round,pad=0,rounding_size={corner_radius}",
                facecolor=color,
                edgecolor="none",
                mutation_scale=1,
                mutation_aspect=1
            )
            ax.add_patch(rect)

    # 3. 坐标轴与刻度文字设置
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    
    # 设置 X 轴文字大小
    ax.set_xticks(np.arange(cols) + 0.5)
    ax.set_xticklabels(df.columns, fontweight='bold', size=label_size)
    
    # 设置 Y 轴文字大小与样式
    ax.set_yticks(np.arange(rows) + 0.5)
    ax.set_yticklabels(df.index[::-1], fontstyle='italic', size=label_size)
    
    # 隐藏边框
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)

    # 4. 添加色条标注 (Colorbar)
    # 创建 ScalarMappable 用于生成色条
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=norm)
    sm.set_array([])
    
    # 调整色条位置和外观
    # shrink 控制高度，aspect 控制粗细，pad 是与主图的距离
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=12, pad=0.08)
    cbar.outline.set_visible(False) # 移除色条外框
    
    # 设置色条上的文字大小和标题
    cbar.ax.tick_params(labelsize=label_size - 2)
    cbar.set_label('Z-score', size=label_size, fontweight='bold')
    
    # 根据图示，可以在色条两端手动标注 High/Low
    cbar.ax.set_yticklabels([f'Low', '', '', '', f'High']) # 简易示例

    plt.tight_layout()
    return fig, ax

# --- 使用示例 ---
data = np.random.randn(12, 4)
genes = [f'Gene_{i}' for i in range(12)]
df_test = pd.DataFrame(data, index=genes, columns=['Liver', 'Muscle', 'Brain', 'Kidney'])

# 调用并设置文字大小为 12
fig, ax = plot_q_heatmap(df_test, label_size=12)
plt.show()

# %%

from matplotlib.gridspec import GridSpec

def plot_fancy_hexbin(x, y, x_label="sepal width (cm)", y_label="sepal length (cm)", title="Hexbin Chart"):
    # 1. 设置全局风格
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # 2. 创建画布与布局 (使用 GridSpec 模拟联合分布图布局)
    fig = plt.figure(figsize=(8, 8), dpi=100)
    gs = GridSpec(4, 4, hspace=0.1, wspace=0.1)

    # 主图 (占据左下角 3x3 区域)
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    # 顶部直方图 (占据第一行)
    ax_hist_x = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    # 右侧直方图 (占据最后三行最右侧)
    ax_hist_y = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    # 3. 绘制主图：六边形分箱图
    # gridsize 控制六边形大小，数值越大蜂窝越小，cmap 使用红色系渐变
    hb = ax_main.hexbin(x, y, gridsize=20, cmap='Blues', mincnt=1, edgecolors='none')
    
    # 4. 绘制边际图：直方图
    # 顶部 X 轴分布
    ax_hist_x.hist(x, bins=20, color="#8ED1F2", edgecolor='white', linewidth=0.8, alpha=0.8)
    # 右侧 Y 轴分布 (orientation='horizontal')
    ax_hist_y.hist(y, bins=20, color='#8ED1F2', edgecolor='white', linewidth=0.8, alpha=0.8, orientation='horizontal')

    # 5. 美化处理
    # 隐藏边际图的坐标轴刻度，只保留主图刻度
    ax_hist_x.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False, 
                          labelbottom=False, labelleft=False)
    ax_hist_y.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False, 
                          labelbottom=False, labelleft=False)
    
    # 去掉边际图的边框
    for ax in [ax_hist_x, ax_hist_y]:
        for spine in ax.spines.values():
            spine.set_visible(False)

    # 设置主图标签
    ax_main.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax_main.set_ylabel(y_label, fontsize=12, fontweight='bold')
    
    # 移除主图多余边框 (Top & Right)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)

    # 6. 添加颜色条 (可选，增加专业性)
    cax = fig.add_axes([0.95, 0.25, 0.02, 0.4]) # [left, bottom, width, height]
    plt.colorbar(hb, cax=cax, label='Count')

    plt.suptitle(title,fontsize=16, fontweight='bold') # 标题位置参考原图
    
    return fig

# --- 模拟数据生成 (以经典的鸢尾花数据分布为参考) ---
np.random.seed(42)
mean = [3.0, 6.0]
cov = [[0.2, 0.1], [0.1, 0.5]]
x, y = np.random.multivariate_normal(mean, cov, 200).T

# 调用绘图
fig = plot_fancy_hexbin(x, y)
plt.show()

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%



