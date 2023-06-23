import random
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
    mostRecentIndex = df_rolling[(df_rolling['TEAM_ID_home'] == homeTeam) & (df_rolling['TEAM_ID_away'] == awayTeam)]['GAME_DATE_EST'].idxmax()
    test_set = df_rolling.loc[mostRecentIndex]
    homeTeamWin = predictWinner(test_set, df_rolling, predictors)

    if homeTeamWin==1:
        return homeTeam
    else:
        return awayTeam


def main():

    startDate = '2021-09-18'
    endDate = '2022-04-11'

    df = pd.read_csv('games.csv')
    df["GAME_DATE_EST"] = pd.to_datetime(df["GAME_DATE_EST"])
    df = df[df['GAME_DATE_EST']<endDate]

    df_regSeason= df[(df['GAME_DATE_EST']<endDate) & (df['GAME_DATE_EST']>startDate)]
    
    #Determine the top 16 teams (8 from each conference)
    NBA_IDS = [1610612737, 1610612738, 1610612739, 1610612740, 1610612741, 1610612742, 1610612743, 1610612744, 1610612745, 1610612746,
                    1610612747, 1610612748, 1610612749, 1610612750, 1610612751, 1610612752, 1610612753, 1610612754, 1610612755, 1610612756,
                    1610612757, 1610612758, 1610612759, 1610612760, 1610612761, 1610612762, 1610612763, 1610612764, 1610612765, 1610612766.]
    

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

    homeTeam_Wins = df_regSeason.groupby('HOME_TEAM_ID')['HOME_TEAM_WINS'].sum()
    awayTeam_Wins = df_regSeason.groupby('VISITOR_TEAM_ID')['HOME_TEAM_WINS'].count() - df_regSeason.groupby('VISITOR_TEAM_ID')['HOME_TEAM_WINS'].sum()
    homeTeam_Wins_array = homeTeam_Wins.to_numpy()
    awayTeam_Wins_array = awayTeam_Wins.to_numpy()
    total_Wins = homeTeam_Wins_array + awayTeam_Wins_array

    All_teams_scores = dict(zip(NBA_IDS, total_Wins))
    All_teams_scores = sorted(All_teams_scores.items(), key=lambda x: x[1], reverse=True)

    All_team_ids = [team_id for team_id, _ in All_teams_scores]

    #Find top 8 teams in each conference
    j=0
    k=0
    WC_top8= []
    EC_top8= []
     
    for id in All_team_ids:
        if (id in WC_team_ID) and (len(WC_top8) < 8):
            WC_top8.append(id)
        elif (id in EC_team_ID) and (len(EC_top8) < 8):
            EC_top8.append(id)

        j+=1
        k+=1

    for element in WC_top8:
        print(WC_teams_ids[element])
    
    for element in EC_top8:
        print(EC_teams_ids[element])

    n_estimators = random.randint(10, 100) 
    min_samples_split = random.randint(2, 20)
    rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split)
    
    predictors = ['TEAM_ID_home', 'TEAM_ID_away']
    columns = ['PTS_home', 'PTS_away', 'FG_PCT_home', 'FG_PCT_away', 'FG3_PCT_home', 'FG3_PCT_away']
    newColumns = [f"{c}_rolling" for c in columns]  

    df_rolling = df_regSeason.groupby("TEAM_ID_home").apply(lambda x: rolling_averages(x, columns, newColumns))
    df_rolling = df_rolling.droplevel('TEAM_ID_home')
    df_rolling.index = range(df_rolling.shape[0])

    #Matches

    #Western Conference

    WC_top4= []
    WC_top2=[]

    #1st round

    WC_top4.append(match(WC_top8[0], WC_top8[7],df_rolling,predictors + newColumns))
    WC_top4.append(match(WC_top8[3], WC_top8[4],df_rolling,predictors + newColumns))
    WC_top4.append(match(WC_top8[2], WC_top8[5],df_rolling,predictors + newColumns))
    WC_top4.append(match(WC_top8[1], WC_top8[6],df_rolling,predictors + newColumns))

    #2nd round

    WC_top2.append(match(WC_top4[0], WC_top4[1],df_rolling,predictors + newColumns))
    WC_top2.append(match(WC_top4[2], WC_top4[3],df_rolling,predictors + newColumns))

    #Conference Final

    WC_winner = match(WC_top2[0], WC_top2[1],df_rolling,predictors + newColumns)

    #Eastern Conference

    EC_top4= []
    EC_top2=[]

    #1st round

    EC_top4.append(match(EC_top8[0], EC_top8[7],df_rolling,predictors + newColumns))
    EC_top4.append(match(EC_top8[3], EC_top8[4],df_rolling,predictors + newColumns))
    EC_top4.append(match(EC_top8[2], EC_top8[5],df_rolling,predictors + newColumns))
    EC_top4.append(match(EC_top8[1], EC_top8[6],df_rolling,predictors + newColumns))

    #2nd round

    EC_top2.append(match(EC_top4[0], EC_top4[1],df_rolling,predictors + newColumns))
    EC_top2.append(match(EC_top4[2], EC_top4[3],df_rolling,predictors + newColumns))

    #Conference Final

    EC_winner = match(EC_top2[0], EC_top2[1],df_rolling,predictors + newColumns)

    #Finals
    Winner = match(WC_winner, EC_winner,df_rolling,predictors + newColumns)


    #Print results
    for id in WC_top4:
        print(WC_teams_ids[id])
    for id in EC_top4:
        print(EC_teams_ids[id])

    for id in WC_top2:
        print(WC_teams_ids[id])
    for id in EC_top2:
        print(EC_teams_ids[id])

    print(WC_teams_ids[WC_winner])
    print(EC_teams_ids[EC_winner])

    if Winner in WC_team_ID:
        print(WC_teams_ids[Winner])
    else:
        print(EC_teams_ids[Winner])





main()