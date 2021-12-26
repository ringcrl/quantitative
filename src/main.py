'''
- rsrs_score 修正标准分 = zscore * r2，效果最佳

- 缩量上涨还会上涨
- 缩量下跌还会下跌
- 高位放巨量上涨必会下跌
- 低位放巨量上涨必会回调
- 低位放巨量下跌必会反弹
- 放量滞涨顶部信号
- 缩量不跌底部信号
- 量大成头，量小成底
- 顶部无量下跌后市还会创新高
- 顶部放量下跌后市很难创新高
- 小跌后的大跌买入
- 看板块一开盘上涨，后面被大盘带下去的
- 寻找稳量上涨的，放量时候出局
'''

#-*- coding:utf-8 -*-
from datetime import datetime, timedelta
from futu import *
import time
import numpy as np
import math
from multiprocessing import Pool
import re 

# 日期计算
current_dt = time.strftime("%Y-%m-%d", time.localtime())
current_date = datetime.strptime(current_dt, '%Y-%m-%d')
previous_date = current_date - timedelta(days = 1)
curr_hour = time.localtime()[3]

freq = 'K_DAY' # K_DAY | K_60M

# 自选股
custom_stocks = [
    'US.QQQ', # 纳指
    'US.DIA', # 道指
    'US.SPY', # 标普

    'HK.00700',
    'US.AMC',

    'US.TSLA',
    'US.NVDA',
    'US.AAPL',
    'US.GOOGL',
    'US.BABA',
]
win_num = 0 # 统计获胜比例

# 动量轮动参数
N = 18 # 计算最新斜率 slope，拟合度 r2 参考最近 N 天，18
M = 200 # 计算最新标准分 zscore，rsrs_score 参考最近 M 天，600
score_threshold = 0.4 # rsrs 标准分指标阈值
mean_day = 50 # 计算结束值，参考最近 mean_day
mean_diff_day = 5 # 计算初始值，参考(mean_day + mean_diff_day)天前，窗口为 mean_diff_day 的一段时间
recall_days = 250 # 回测天数

# 择时模块-计算综合信号，rsrs 信号算法参考优化说明，与其他值共同判断减少误差
def get_timing_signal(stock_data, slope_series):
    # 返回格式：stock_signal, rsrs_score, val_status

    curr_rsrs_score = get_rsrs_score(stock_data, slope_series) # number
    volume_signal = get_volume_signal(stock_data) # VAL_UP | VAL_DOWN | VAL_STILL
    shooting_signal = get_shooting_signal(stock_data) # None | TOP | BOTTOM
    stock_score = get_stock_score(stock_data)
    op = 'HOLD' # BUY | SELL | HOLD

    if curr_rsrs_score >= score_threshold and volume_signal == 'VAL_UP':
        op = 'BUY'
    elif curr_rsrs_score >= score_threshold and volume_signal == 'VAL_DOWN':
        op = 'SELL'
    elif curr_rsrs_score <= -score_threshold and volume_signal == 'VAL_DOWN':
        op = 'SELL'
    elif curr_rsrs_score <= -score_threshold and volume_signal == 'VAL_UP':
        op = 'BUY'
    else:
        op = 'HOLD'

    return {
        "op": op,
        "rsrs_score": curr_rsrs_score,
        "volume_signal": volume_signal,
        "shooting_signal": shooting_signal,
        "stock_score": stock_score,
    }

# 计算 rsrs 信号：https://zhuanlan.zhihu.com/p/33501881
def get_rsrs_score(stock_data, slope_series):
    high_low_data = stock_data[-N:]
    intercept, slope, r2 = get_ols(high_low_data.low, high_low_data.high)
    slope_series.append(slope)
    rsrs_score = get_zscore(slope_series[-M:]) * r2
    return rsrs_score

# 成交量信号
def get_volume_signal(stock_data):
    mean_day_value = stock_data.turnover[-mean_day:].mean()
    mean_diff_day_value = stock_data.turnover[-mean_diff_day:].mean()
    if mean_diff_day_value > mean_day_value:
        return 'VAL_UP'
    else:
        return 'VAL_DOWN'

# 射击之星和锤头线信号
def get_shooting_signal(stock_data):
    # 返回格式：None | 'TOP' | 'BOTTOM'
    num = 2
    len_rate = 2 # 一般2-3倍
    top_num = 0
    button_num = 0
    latest_data = stock_data[-num:]
    for i in range(num):
        close_price = latest_data.close.values[i]
        high_price = latest_data.high.values[i]
        low_price = latest_data.low.values[i]
        open_price = latest_data.open.values[i]
        entity_len = abs(close_price - open_price)
        upline_len = 0
        downline_len = 0
        # 收盘价高于开盘价
        if close_price >= open_price:
            upline_len = abs(high_price - close_price)
            downline_len = abs(low_price - open_price)
        else:
            upline_len = abs(high_price - open_price)
            downline_len = abs(low_price - close_price)
        if upline_len >= entity_len * len_rate:
            top_num += 1
        if downline_len >= entity_len * len_rate:
            button_num += 1
    
    if top_num == num:
        return 'TOP'
    elif button_num == num:
        return 'BOTTOM'
    else:
        return None

# 根据市值，获取股票池
def get_market_cap(market_val=50000000000):
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
        ret, data = quote_ctx.get_stock_filter(market=Market.US, filter_list=simple_filter, begin=nBegin)  # 对香港市场的股票做简单和财务筛选
        if ret == RET_OK:
            last_page, all_count, ret_list = data
            print('all count = ', all_count)
            
            for item in ret_list:
                stock_list.append(item.stock_code)
        else:
            print('error: ', data)
        time.sleep(1)

    quote_ctx.close()
    return stock_list

# 根据股票代码获取历史K数据
def get_code_data(code, freq=freq, count=1000):
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
        quote_ctx.close()
        print(err_message)
        return None

# 获取股票得分
def get_stock_score(stock_data):
        # 收盘价
    y = stock_data['log'] = np.log(stock_data.close)
    # 分析的数据个数（天）
    x = stock_data['num'] = np.arange(stock_data.log.size)
    # 拟合 1 次多项式
    # y = kx + b, slope 为斜率 k，intercept 为截距 b
    slope, intercept = np.polyfit(x, y, 1)
    # (e ^ slope) ^ 250 - 1
    annualized_returns = math.pow(math.exp(slope), 250) - 1
    r_squared = 1 - (sum((y - (slope * x + intercept))**2) / ((len(y) - 1) * np.var(y, ddof=1)))
    score = annualized_returns * r_squared
    return score

# 择时模块-计算线性回归统计值
# 对输入的自变量每日最低价 x(series) 和因变量每日最高价 y(series) 建立 OLS 回归模型,返回元组(截距,斜率,拟合度)
# R2 统计学线性回归决定系数，也叫判定系数，拟合优度。
# R2 范围 0 ~ 1，拟合优度越大，自变量对因变量的解释程度越高，越接近 1 越好。
# 公式说明： https://blog.csdn.net/snowdroptulip/article/details/79022532
#           https://www.cnblogs.com/aviator999/p/10049646.html
def get_ols(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    r2 = 1 - (sum((y - (slope * x + intercept))**2) / ((len(y) - 1) * np.var(y, ddof=1)))
    return (intercept, slope, r2)

# 择时模块-计算标准分，通过斜率列表计算并返回截至回测结束日的最新标准分
def get_zscore(slope_series):
    mean = np.mean(slope_series)
    std = np.std(slope_series)
    return (slope_series[-1] - mean) / std

# 择时模块-设定初始斜率序列，通过前 M 日最高最低价的线性回归计算初始的斜率，返回斜率的列表
def initial_slope_series(general_stock, N, M):
    return [get_ols(general_stock.low[i:i+N], general_stock.high[i:i+N])[1] for i in range(M)]

# 获取股票信号
def get_stock_signals(stock_data):
    slope_series = initial_slope_series(stock_data, N, M)[:-1]
    return get_timing_signal(stock_data, slope_series)

# 运行回测
def recall(stock_code):
    monkey_base = 100000
    monkey_count = monkey_base
    stock_count = 0
    stock_data_all = get_code_data(stock_code)
    if len(stock_data_all) == 0:
        return f'''{stock_code} error'''
    max_retreat = 0 # 最大回撤百分比
    before_total = 0 # 当前总额
    keep_stocks = 0 # 用于比对的股数，一开始躺平不动
    trade_num = 0 # 交易次数

    for i in range(recall_days):
        before_day = recall_days - i
        stock_data = stock_data_all[-before_day-N-M:-before_day]
        signals = get_stock_signals(stock_data)
        close_price = stock_data.close.values[-1]
        is_buy = signals['op'] == 'BUY'
        is_sell = signals['op'] == 'SELL'
        info = f'''{stock_code} {stock_data.time_key.values[-1][:10]} {signals['op']} {format(signals['rsrs_score'], '.2f')} {signals['volume_signal']} {format(close_price, '.2f')}'''

        if i == 0:
            keep_stocks = monkey_count / close_price

        if is_buy and monkey_count != 0:
            stock_count = monkey_count / close_price
            monkey_count = 0
            before_total = stock_count * close_price
            trade_num += 1
            print(info + ' 买入')
        elif is_sell and stock_count != 0:
            monkey_count = stock_count * close_price
            stock_count = 0
            # 当笔交易亏钱
            if monkey_count < before_total:
                curr_retreat = (before_total - monkey_count) / before_total * 100
                if curr_retreat > max_retreat:
                    max_retreat = curr_retreat
            trade_num += 1
            print(info + ' 卖出')
        else:
            print(info)
    final_close = stock_data_all.close.values[-1]
    total = monkey_count + stock_count * final_close
    keep_total = keep_stocks * final_close
    op_res = (total - monkey_base) / monkey_base * 100
    no_op_res = (keep_total - monkey_base) / monkey_base * 100
    is_win = op_res > no_op_res and op_res > 0
    res = f'''{stock_code} 盈亏比：{format(op_res, '.2f')}% 躺平盈亏比：{format(no_op_res, '.2f')}% 最新单价：{final_close} 最大回撤：-{format(max_retreat, '.2f')}% 交易次数：{trade_num} 获胜：{is_win}'''
    print(res)
    return res

# 批量运行回测
def batch_recall():
    p = Pool(len(custom_stocks))
    res = p.map(recall, custom_stocks)    
    p.close()
    p.join()
    info = '\r\n'.join(res)
    win_len = len(re.findall('True', info))
    msg = f'''
{current_dt} 近{recall_days}天回测结果 胜率：{format(win_len / len(custom_stocks) * 100, '.2f')}%
{info}
'''
    print(msg)

# 批量获取自选股信号
def batch_op_signal(is_custom=True):
    stocks = []
    if is_custom:
        stocks = custom_stocks
    else:
        stocks  = get_market_cap()

    p = Pool(len(stocks))
    res = p.map(op_signal, stocks)
    p.close()
    p.join()
    info = '\r\n'.join(res)
    msg = f'''
{current_dt} 操作信号
{info}
'''
    print(msg)


def op_signal(stock_code):
    stock_data = get_code_data(stock_code)

    # 开盘中去掉今天的数据，防止成交量干扰
    if curr_hour >= 22 or curr_hour < 5:
        stock_data = stock_data[:-1]

    before_signals = get_stock_signals(stock_data[:-1])
    curr_signals = get_stock_signals(stock_data)
    prefix = ''
    if before_signals['op'] != curr_signals['op']:
        if curr_signals['op'] == 'BUY':
            prefix = '买 '
        elif curr_signals['op'] == 'SELL':
            prefix = '卖 '
        else:
            prefix = '观察 '
    res = f'''{prefix}{stock_code} 【{before_signals['op']}->{curr_signals['op']}】 【{format(curr_signals['rsrs_score'], '.2f')} {curr_signals['volume_signal']}】'''
    return res

if __name__ == "__main__":
    # batch_recall()
    batch_op_signal()

