from flask import Flask, render_template
import pandas as pd
from visualizations.plotly_charts import (
    create_pie_chart,
    create_line_chart,
    create_comparison_charts,
    create_youtube_sentiment_box,
    create_news_sentiment_bar,
    create_stacked_area_chart,
    create_facet_sentiment_chart,
    create_combined_stacked_sentiment,
    create_news_sentiment_scatter,
    create_combined_treemap,
    create_source_sentiment_comparison,
    generate_wordcloud_by_sentiment,
    create_source_sentiment_comparison
)

app = Flask(__name__)

# Load main dataset
df = pd.read_csv('static/data/CombinedData1.csv')
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

# Load new combined dataset
combined_df = pd.read_csv('static/data/Combined.csv')
combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')

# Extract YouTube data safely if exists
youtube_df = df[df['source'].str.lower() == 'youtube'].copy()
youtube_df['date'] = pd.to_datetime(youtube_df['date'], errors='coerce', dayfirst=True)


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/twitter')
def twitter_analysis():
    twitter_df = df[df['source'].str.lower() == 'twitter']
    pie = create_pie_chart(twitter_df)
    line = create_line_chart(twitter_df)
    return render_template('twitter.html', pie_chart=pie, line_chart=line)


@app.route('/news')
def news_analysis():
    news_df = df[df['source'].str.lower() == 'news']
    pie = create_pie_chart(news_df)
    bar = create_news_sentiment_bar(df)
    return render_template('news.html', pie_chart=pie, sentiment_chart=bar)

@app.route('/youtube')
def youtube_analysis():
    pie = create_pie_chart(youtube_df)
    box = create_youtube_sentiment_box(df)
    return render_template('youtube.html', pie_chart=pie, line_chart=box)
@app.route('/comparison')
def comparison_analysis():
    # Existing charts
    bar, line = create_comparison_charts(df)
    stacked = create_stacked_area_chart(df)
    facet = create_facet_sentiment_chart(df)

    # 🔁 Combine datasets
    merged_df = pd.concat([df, combined_df])  # Rename to avoid overwriting 'combined_df'

    # ✅ New source-sentiment chart
    source_sentiment_chart = create_source_sentiment_comparison(merged_df)

    return render_template(
        'comparison.html',
        pie_chart=bar,
        line_chart=line,
        stacked_chart=stacked,
        facet_chart=facet,
        source_sentiment_chart=source_sentiment_chart  # Pass to template
    )



@app.route('/combined')
def combined_analysis():
    pie = create_pie_chart(combined_df)
    line = create_line_chart(combined_df)
    wc_positive = generate_wordcloud_by_sentiment(combined_df, 'positive')
    wc_negative = generate_wordcloud_by_sentiment(combined_df, 'negative')
    wc_neutral = generate_wordcloud_by_sentiment(combined_df, 'neutral')
    
    stacked = create_combined_stacked_sentiment(combined_df)

    return render_template(
        'combined.html',
        pie_chart=pie,
        line_chart=line,
        wc_positive=wc_positive,
        wc_negative=wc_negative,
        wc_neutral=wc_neutral,
        stacked_chart=stacked,
        treemap = create_combined_treemap(combined_df)
    )



if __name__ == '__main__':
    app.run(debug=True)
