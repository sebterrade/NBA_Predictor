import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

plt.style.use('ggplot')



global rf 
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

global gameNum


def rolling_averages(team, columns, newColumns):
    team = team.sort_values("GAME_DATE_EST")
    rolling_stats = team[columns].rolling(3, closed='left').mean()
    team[newColumns] = rolling_stats
    team = team.dropna(subset=newColumns)
    return team

# def make_predictions(data, predictors):
#     train_set = data[(data['GAME_DATE_EST']>'2021-09-18') &  (data['GAME_DATE_EST']<'2022-04-11')]
#     test_set = data[(data['GAME_DATE_EST']>'2022-04-11') & (data['GAME_DATE_EST']<'2022-06-30')]
#     # test_set =np.array([[1610612737,1610612741]]) 
#     rf.fit(train_set[predictors], train_set['HOME_TEAM_WINS'])
#     predictions = rf.predict(test_set[predictors])
#     combined = pd.DataFrame(dict(actual=test_set['HOME_TEAM_WINS'], predicted=predictions), index = test_set.index)
#     precision = precision_score(test_set['HOME_TEAM_WINS'], predictions)
#     return combined, precision

def predictWinner( test_set, data, predictors):
    train_set = data
    rf.fit(train_set[predictors], train_set['HOME_TEAM_WINS'])
    test_set = test_set[predictors].values.reshape(1,-1)
    predictedOutcome = rf.predict(test_set)
    return predictedOutcome

def match(homeTeam, awayTeam, df_rolling, predictors):
    mostRecentIndex = df_rolling[(df_rolling['HOME_TEAM_ID'] == homeTeam) & (df_rolling['VISITOR_TEAM_ID'] == awayTeam)]['GAME_DATE_EST'].idxmax()
    test_set = df_rolling.loc[mostRecentIndex]
    homeTeamWin = predictWinner(test_set, df_rolling, predictors)

    return homeTeamWin


def main():

    startDate = '2021-09-18'
    endDate = '2022-04-11'
    
    WC_team_ID = [1610612742, 1610612743,  1610612744, 1610612745, 1610612746, 1610612747, 1610612763,1610612750, 1610612740, 
                  1610612760, 1610612756, 1610612757, 1610612758, 1610612759, 1610612762]
    
    WC_NBA_teams = [ 'Dallas Mavericks', 'Denver Nuggets',  'Golden State Warriors', 'Houston Rockets', 'LA Clippers', 
                    'Los Angeles Lakers', 'Memphis Grizzlies', 'Minnesota Timberwolves', 'New Orleans Pelicans', 'Oklahoma City Thunder', 'Phoenix Suns', 
                 'Portland Trail Blazers', 'Sacramento Kings', 'San Antonio Spurs', 'Utah Jazz']
    
    EC_team_ID = [1610612737, 1610612738, 1610612751, 1610612766, 1610612741, 1610612739, 1610612754, 1610612748, 1610612749, 
                  1610612752, 1610612753,  1610612755, 1610612765, 1610612761, 1610612764]

    EC_NBA_teams = ['Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets', 'Chicago Bulls', 'Cleveland Cavaliers', 'Indiana Pacers', 'Miami Heat', 
                    'Milwaukee Bucks', 'New York Knicks' ,'Orlando Magic', 'Philadelphia 76ers', 'Detroit Pistons', 'Toronto Raptors', 'Washington Wizards']
    
    WC_teams_ids = dict(zip(WC_team_ID,WC_NBA_teams))
    EC_teams_ids = dict(zip(EC_team_ID,EC_NBA_teams))

    homeTeam_Wins = df.groupby('HOME_TEAM_ID')['HOME_TEAM_WINS'].sum()
    awayTeam_Wins = df.groupby('VISITOR_TEAM_ID')['HOME_TEAM_WINS'].count() - df.groupby('VISITOR_TEAM_ID')['HOME_TEAM_WINS'].sum()
    homeTeam_Wins_array = homeTeam_Wins.to_numpy()
    awayTeam_Wins_array = awayTeam_Wins.to_numpy()
    total_Wins = homeTeam_Wins_array + awayTeam_Wins_array
    



    df = pd.read_csv('games.csv')
    df["GAME_DATE_EST"] = pd.to_datetime(df["GAME_DATE_EST"])
    df = df[df['GAME_DATE_EST']<endDate]

    df_regSeason= df[(df['GAME_DATE_EST']<endDate) & (df['GAME_DATE_EST']>startDate)]

    #Determine the top 

 
    rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
    
    predictors = ['TEAM_ID_home', 'TEAM_ID_away']
    columns = ['PTS_home', 'PTS_away', 'FG_PCT_home', 'FG_PCT_away', 'FG3_PCT_home', 'FG3_PCT_away']
    newColumns = [f"{c}_rolling" for c in columns]  

    df_rolling = df_regSeason.groupby("TEAM_ID_home").apply(lambda x: rolling_averages(x, columns, newColumns))
    df_rolling = df_rolling.droplevel('TEAM_ID_home')
    df_rolling.index = range(df_rolling.shape[0])

    gameNum=0

    match_outcome = match(1610612749, 1610612744,df_rolling,predictors + newColumns)
    print(match_outcome)


main()