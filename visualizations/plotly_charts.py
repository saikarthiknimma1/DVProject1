import plotly.express as px
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def create_pie_chart(df, title='Sentiment Distribution'):
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    fig = px.pie(sentiment_counts, names='sentiment', values='count', title=title)
    return fig.to_html(full_html=False)

def create_line_chart(df, title='Sentiment Over Time'):
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['date'])
    df['date_only'] = df['date'].dt.date
    daily_counts = df.groupby('date_only').size().reset_index(name='count')
    fig = px.line(daily_counts, x='date_only', y='count', title=title)
    fig.update_traces(mode='lines+markers')
    fig.update_layout(xaxis_title='Date', yaxis_title='Mentions')
    return fig.to_html(full_html=False)

def create_comparison_charts(df):
    sentiment_by_source = df.groupby(['source', 'sentiment']).size().reset_index(name='count')
    bar_chart = px.bar(sentiment_by_source, x='source', y='count', color='sentiment', title='Sentiment Distribution by Source', barmode='group')
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['date'])
    df['date_only'] = df['date'].dt.date
    time_chart_data = df.groupby(['source', 'date_only', 'sentiment']).size().reset_index(name='count')
    time_chart = px.line(time_chart_data, x='date_only', y='count', color='source', line_dash='sentiment', title='Time-series Sentiment by Source')
    time_chart.update_traces(mode='lines+markers')
    return bar_chart.to_html(full_html=False), time_chart.to_html(full_html=False)

def create_stacked_area_chart(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['date'])
    df['date_only'] = df['date'].dt.date
    grouped = df.groupby(['date_only', 'source']).size().reset_index(name='count')
    fig = px.area(grouped, x='date_only', y='count', color='source', title='Overall Mentions Over Time by Source')
    fig.update_layout(xaxis_title='Date', yaxis_title='Total Mentions')
    return fig.to_html(full_html=False)

def create_facet_sentiment_chart(df):
    sentiment_by_source = df.groupby(['sentiment', 'source']).size().reset_index(name='count')
    fig = px.bar(sentiment_by_source, x='source', y='count', color='source', facet_col='sentiment', title='Sentiment Comparison by Source (Faceted View)')
    return fig.to_html(full_html=False)

def top_keywords_by_sentiment(df, n=10):
    from sklearn.feature_extraction.text import CountVectorizer
    results = {}
    for sentiment in ['positive', 'negative', 'neutral']:
        texts = df[df['sentiment'] == sentiment]['text'].dropna().astype(str)
        vectorizer = CountVectorizer(stop_words='english', max_features=n)
        X = vectorizer.fit_transform(texts)
        keywords = vectorizer.get_feature_names_out()
        frequencies = X.toarray().sum(axis=0)
        keywords_df = pd.DataFrame({'keyword': keywords, 'count': frequencies})
        fig = px.bar(keywords_df, x='keyword', y='count', title=f'Top {n} Keywords for {sentiment.capitalize()} Sentiment')
        results[sentiment] = fig.to_html(full_html=False)
    return results

def create_news_top_keywords(df, n=10):
    from sklearn.feature_extraction.text import CountVectorizer
    news_df = df[df['source'].str.lower() == 'news'].copy()
    results = {}
    for sentiment in ['positive', 'negative', 'neutral']:
        texts = news_df[news_df['sentiment'] == sentiment]['text'].dropna().astype(str)
        if texts.empty:
            continue
        try:
            vectorizer = CountVectorizer(stop_words='english', max_features=n)
            X = vectorizer.fit_transform(texts)
            if X.shape[1] == 0:
                continue
            keywords = vectorizer.get_feature_names_out()
            frequencies = X.toarray().sum(axis=0)
            keywords_df = pd.DataFrame({'keyword': keywords, 'count': frequencies})
            fig = px.bar(keywords_df, x='keyword', y='count', title=f'Top {n} Keywords for {sentiment.capitalize()} Sentiment in News')
            results[sentiment] = fig.to_html(full_html=False)
        except:
            continue
    if not results:
        fallback_df = pd.DataFrame({'keyword': ['no keywords'], 'count': [1]})
        fig = px.bar(fallback_df, x='keyword', y='count', title='No Keyword Data Available for News')
        results['info'] = fig.to_html(full_html=False)
    return results

def create_youtube_sentiment_box(df):
    youtube_df = df[df['source'].str.lower() == 'youtube'].copy()
    youtube_df['text_length'] = youtube_df['text'].astype(str).apply(len)
    fig = px.box(youtube_df, x='sentiment', y='text_length', color='sentiment', title='YouTube Text Length Distribution by Sentiment')
    return fig.to_html(full_html=False)

def create_news_sentiment_bar(df):
    news_df = df[df['source'].str.lower() == 'news'].copy()
    sentiment_counts = news_df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    fig = px.bar(sentiment_counts, x='count', y='sentiment', orientation='h', color='sentiment', title='News Sentiment Breakdown')
    return fig.to_html(full_html=False)

def create_news_sentiment_scatter(df):
    news_df = df[df['source'].str.lower() == 'news'].copy()
    news_df['text_length'] = news_df['text'].astype(str).apply(len)
    fig = px.scatter(news_df, x='text_length', y=news_df.index, color='sentiment', title='News Sentiment vs. Text Length', labels={'text_length': 'Text Length', 'index': 'Article Index'}, opacity=0.6)
    return fig.to_html(full_html=False)

def create_news_sentiment_over_time(df):
    news_df = df[df['source'].str.lower() == 'news'].copy()
    news_df['date'] = pd.to_datetime(news_df['date'], dayfirst=True, errors='coerce')
    news_df = news_df.dropna(subset=['date'])
    news_df['date_only'] = news_df['date'].dt.date
    trend = news_df.groupby(['date_only', 'sentiment']).size().reset_index(name='count')
    fig = px.line(trend, x='date_only', y='count', color='sentiment', title='News Sentiment Over Time')
    fig.update_traces(mode='lines+markers')
    fig.update_layout(xaxis_title='Date', yaxis_title='Mentions')
    return fig.to_html(full_html=False)

def generate_wordcloud_by_sentiment(df, sentiment):
    text_data = df[df['sentiment'] == sentiment]['text'].dropna().astype(str).str.cat(sep=' ')
    if not text_data.strip():
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
    buffer = BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"

def create_combined_stacked_sentiment(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df[df['date'].dt.year <= 2007]  # 👈 Filter only up to 2008
    df['date_only'] = df['date'].dt.date

    sentiment_counts = df.groupby(['date_only', 'sentiment']).size().reset_index(name='count')

    fig = px.area(
        sentiment_counts,
        x='date_only',
        y='count',
        color='sentiment',
        title='Combined Dataset Sentiment Trends Over Time (until 2007)'
    )
    fig.update_layout(xaxis_title='Date', yaxis_title='Mentions')
    return fig.to_html(full_html=False)

def create_combined_treemap(df):
    df['sentiment'] = df['sentiment'].str.lower()
    df['source'] = df['source'].str.lower()

    grouped = df.groupby(['sentiment', 'source']).size().reset_index(name='count')
    grouped = grouped.dropna(subset=['sentiment', 'source'])

    print("Grouped Treemap Data:\n", grouped)

    if grouped['source'].nunique() == 1:
        fig = px.treemap(
            grouped,
            path=['sentiment'],
            values='count',
            title='Treemap of Sentiment'
        )
    else:
        fig = px.treemap(
            grouped,
            path=['sentiment', 'source'],
            values='count',
            title='Treemap of Sentiment by Source'
        )

    return fig.to_html(full_html=False)

def create_source_sentiment_comparison(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['date_only'] = df['date'].dt.date

    # Create a combined label for legend
    df['sentiment_source'] = df['sentiment'].str.lower() + "_" + df['source'].str.lower()

    grouped = df.groupby(['date_only', 'sentiment_source']).size().reset_index(name='count')

    fig = px.line(grouped, x='date_only', y='count', color='sentiment_source',
                  title='Sentiment Comparison: Twitter, News & YouTube')
    fig.update_traces(mode='lines')
    fig.update_layout(xaxis_title='Date', yaxis_title='Mentions')

    return fig.to_html(full_html=False)
def create_source_sentiment_comparison(df):
    # Parse and clean date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['date_only'] = df['date'].dt.date

    # Filter data to before or in 2011
    df = df[df['date'] <= pd.to_datetime('2011-12-31')]

    # Combine sentiment and source for easier comparison
    df['sentiment_source'] = df['sentiment'].str.lower() + '_' + df['source'].str.lower()

    # Group by date and sentiment_source
    trend = df.groupby(['date_only', 'sentiment_source']).size().reset_index(name='count')

    # Clip high values (optional to prevent spike distortion)
    trend = trend[trend['count'] <= 1000]

    # Plot
    fig = px.line(
        trend,
        x='date_only',
        y='count',
        color='sentiment_source',
        title='Sentiment Comparison: Twitter, News & YouTube',
        labels={'date_only': 'Date', 'count': 'Mentions'},
    )

    # Style updates
    fig.update_layout(
        template='plotly_white',
        yaxis=dict(range=[0, 200]),  # Restrict Y-axis for better clarity
        legend_title='Sentiment + Source',
        xaxis_title='Date',
        yaxis_title='Mentions'
    )

    return fig.to_html(full_html=False)