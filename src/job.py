'''
优化说明:
    1.使用修正标准分
        rsrs_score的算法有：
            仅斜率slope，效果一般；
            仅标准分zscore，效果不错；
            修正标准分 = zscore * r2，效果最佳;
            右偏标准分 = 修正标准分 * slope，效果不错。
    2.将原策略的每次持有两只etf改成只买最优的一个，收益显著提高
    3.将每周调仓换成每日调仓，收益显著提高
    4.因为交易etf，所以手续费设为万分之三，印花税设为零，未设置滑点
    5.修改股票池中候选etf，删除银行，红利等收益较弱品种，增加纳指etf以增加不同国家市场间轮动的可能性
    6.根据研报，默认参数介已设定为最优
    7.加入防未来函数
    8.增加择时与选股模块的打印日志，方便观察每笔操作依据
'''

#-*- coding:utf-8 -*-
from datetime import datetime, timedelta
from futu import *
import time
import numpy as np
import math
from sendmail import mail

# 自选股票池
custom_stock_pool = [
    'HK.00700', # 腾讯
    'HK.09988', # 阿里
    'HK.01211', # 比亚迪
]

# 动量轮动参数
stock_num = 1 # 买入评分最高的前 stock_num 只股票
momentum_day = 29 # 最新动量参考最近 momentum_day 的

ref_stock = 'HK.800000' #用 ref_stock 做择时计算的基础数据
N = 18 # 计算最新斜率 slope，拟合度 r2 参考最近 N 天
M = 600 # 计算最新标准分 zscore，rsrs_score 参考最近 M 天
score_threshold = 0.7 # rsrs 标准分指标阈值
# ma 择时参数
mean_day = 20 # 计算结束 ma 收盘价，参考最近 mean_day
mean_diff_day = 3 # 计算初始 ma 收盘价，参考(mean_day + mean_diff_day)天前，窗口为 mean_diff_day 的一段时间

run_today_day = 1
day = 1

send_info = []

def get_stock_pool():
    # return custom_stock_pool
    market_list = get_market_cap()
    res = []
    for market_item in market_list:
        res.append(market_item['stock_code'])
    return res

# 根据市值，获取股票池
def get_market_cap(market_val=1000000000000):
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    simple_filter = SimpleFilter()
    simple_filter.stock_field = StockField.MARKET_VAL
    simple_filter.filter_min = market_val
    simple_filter.is_no_filter = False

    stock_list = list()
    nBegin = 0
    last_page = False
    ret_list = list()
    while not last_page:
        nBegin += len(ret_list)
        ret, data = quote_ctx.get_stock_filter(market=Market.HK, filter_list=simple_filter, begin=nBegin)  # 对香港市场的股票做简单和财务筛选
        if ret == RET_OK:
            last_page, all_count, ret_list = data
            print('all count = ', all_count)
            
            for item in ret_list:
                stock_list.append({'stock_code': item.stock_code, 'stock_name': item.stock_name})
        else:
            print('error: ', data)
        time.sleep(1)

    quote_ctx.close()  # 结束后记得关闭当条连接，防止连接条数用尽
    return stock_list

# 根据股票代码获取历史K数据
def get_code_data(code, freq, count):
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    ret_sub, err_message = quote_ctx.subscribe([code], [freq], subscribe_push=False)
    if ret_sub == RET_OK:
        ret, data = quote_ctx.get_cur_kline(code, count, freq, AuType.QFQ)
        quote_ctx.close()
        if ret == RET_OK:
            return data
        else:
            print(ret)
            return None
    else:
        print(err_message)
        return None

# 1-1 选股模块-动量因子轮动
# 基于股票年化收益和判定系数打分,并按照分数从大到小排名
def get_rank(_stock_pool, day):
    score_list = []
    global send_info
    for stock in _stock_pool:
        current_dt = time.strftime("%Y-%m-%d", time.localtime())
        current_dt = datetime.strptime(current_dt, '%Y-%m-%d')
        previous_date  = current_dt - timedelta(days = day)
        data = get_code_data(stock, 'K_DAY', momentum_day)
        # 收盘价
        y = data['log'] = np.log(data.close)
        # 分析的数据个数（天）
        x = data['num'] = np.arange(data.log.size)
        # 拟合 1 次多项式
        # y = kx + b, slope 为斜率 k，intercept 为截距 b
        slope, intercept = np.polyfit(x, y, 1)
        # (e ^ slope) ^ 250 - 1
        annualized_returns = math.pow(math.exp(slope), 250) - 1
        r_squared = 1 - (sum((y - (slope * x + intercept))**2) / ((len(y) - 1) * np.var(y, ddof=1)))
        score = annualized_returns * r_squared
        score_list.append(score)
    stock_dict = dict(zip(_stock_pool, score_list))
    sort_list = sorted(stock_dict.items(), key = lambda item:item[1], reverse = True)
    for stock in sort_list:
        stock_code = stock[0]
        stock_score = format(stock[1], '.4f') # 分数保留4位小数
        stock_name = stock[0]
        send_info.append(f'''{stock_code} | {stock_score}''')
    code_list = []
    for i in range((len(_stock_pool))):
        code_list.append(sort_list[i][0])
    rank_stock = code_list[0:stock_num]
    return rank_stock

# 2-1 择时模块-计算线性回归统计值
# 对输入的自变量每日最低价 x(series) 和因变量每日最高价 y(series) 建立 OLS 回归模型,返回元组(截距,斜率,拟合度)
# R2 统计学线性回归决定系数，也叫判定系数，拟合优度。
# R2 范围 0 ~ 1，拟合优度越大，自变量对因变量的解释程度越高，越接近 1 越好。
# 公式说明： https://blog.csdn.net/snowdroptulip/article/details/79022532
#           https://www.cnblogs.com/aviator999/p/10049646.html
def get_ols(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    r2 = 1 - (sum((y - (slope * x + intercept))**2) / ((len(y) - 1) * np.var(y, ddof=1)))
    return (intercept, slope, r2)

# 2-2 择时模块-设定初始斜率序列
# 通过前 M 日最高最低价的线性回归计算初始的斜率,返回斜率的列表
def initial_slope_series():
    current_dt = time.strftime("%Y-%m-%d", time.localtime())
    current_dt = datetime.strptime(current_dt, '%Y-%m-%d')
    previous_date  = current_dt - timedelta(days = day)
    # data = jq.get_price(ref_stock, end_date = previous_date, count = N + M, frequency='daily', fields=['high', 'low'])
    
    # 获取 futu 数据
    code = 'HK.00700'
    freq = 'K_DAY'
    futu_data = get_code_data(code, freq, N + M)
    res = [get_ols(futu_data.low[i:i+N], futu_data.high[i:i+N])[1] for i in range(M)]
    return res

# 2-3 择时模块-计算标准分
# 通过斜率列表计算并返回截至回测结束日的最新标准分
def get_zscore(slope_series):
    mean = np.mean(slope_series)
    std = np.std(slope_series)
    return (slope_series[-1] - mean) / std

# 2-4 择时模块-计算综合信号
# 1.获得 rsrs 与 MA 信号,rsrs 信号算法参考优化说明，MA 信号为一段时间两个端点的 MA 数值比较大小
# 2.信号同时为 True 时返回买入信号，同为 False 时返回卖出信号，其余情况返回持仓不变信号
# 解释：
#       MA 信号：MA 指标是英文(Moving average)的简写，叫移动平均线指标。
#       RSRS 择时信号：
#               https://www.joinquant.com/view/community/detail/32b60d05f16c7d719d7fb836687504d6?type=1
def get_timing_signal(stock):
    # 计算 MA 信号
    current_dt = time.strftime("%Y-%m-%d", time.localtime())
    current_dt = datetime.strptime(current_dt, '%Y-%m-%d')
    previous_date  = current_dt - timedelta(days = day)
    # close_data = jq.get_price(ref_stock, end_date = previous_date, count = mean_day + mean_diff_day,  frequency = 'daily',  fields = ['close'])
    # 获取 futu 数据
    close_data = get_code_data(ref_stock, 'K_DAY', mean_day + mean_diff_day)

    # 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1，23 天，要后 20 天
    today_MA = close_data.close[mean_diff_day:].mean()
    # 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0，23 天，要前 20 天
    before_MA = close_data.close[:-mean_diff_day].mean()
    # 计算 rsrs 信号
    # high_low_data = jq.get_price(ref_stock, end_date = previous_date, count = N,  frequency='daily',   fields = ['high', 'low'])
    # 获取 futu 数据
    high_low_data = get_code_data(ref_stock, 'K_DAY', N)

    intercept, slope, r2 = get_ols(high_low_data.low, high_low_data.high)
    slope_series.append(slope)
    rsrs_score = get_zscore(slope_series[-M:]) * r2
    # 综合判断所有信号
    if rsrs_score > score_threshold and today_MA > before_MA:
        return "BUY"
    elif rsrs_score < -score_threshold and today_MA < before_MA:
        return "SELL"
    else:
        return "KEEP"

slope_series = initial_slope_series()[:-1] # 除去回测第一天的 slope ，避免运行时重复加入

def send_email(message):
    ret = 0
    for _ in range(10):
        if ret:
            # 邮件发送成功推出
            break
        else:
            # 没有发送成功或失败继续
            ret = mail(message)
            time.sleep(1)

def run_today():
    current_dt = time.strftime("%Y-%m-%d", time.localtime())
    current_dt = datetime.strptime(current_dt, '%Y-%m-%d')
    message = f"""
===择时信号===    
"""
    stock_pool = get_stock_pool()
    check_out_list = get_rank(stock_pool, run_today_day)
    for each_check_out in check_out_list:
        stock_name = each_check_out
        stock_code = each_check_out
    # 获取综合择时信号
    timing_signal = get_timing_signal(ref_stock)
    if timing_signal == 'SELL':
        message += f'''清仓！卖卖卖!
'''
    else:
        message += f'''{stock_code}
'''
    message += f'''===备选股===
'''
    message += "\r\n".join(send_info)
 
    print(message)
    # send_email(message)


if __name__ == "__main__":
    run_today()
