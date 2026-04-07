# ============================================================
# extract_serve_receive.R
#
# Reads a .dvw file and returns one row per serve-receive
# point containing all context needed for the set-prediction
# model.
#
# Usage:
#   source("extract_serve_receive.R")
#   result <- extract_serve_receive_plays("path/to/file.dvw", focus_team_id = 249)
#   write_csv(result, "data/cleaned_plays.csv")
# ============================================================

library(datavolley)
library(dplyr)
library(readr)

# ============================================================
# extract_serve_receive_plays()
#
# Parameters:
#   dvw_path        : character, path to your .dvw file
#   focus_team_id   : numeric team ID of the team you are
#                     analyzing (determines sign of score_diff)
#   focus_team_name : name of team you are analyzing
# ============================================================

extract_serve_receive_plays <- function(dvw_path, focus_team_id, focus_team_name) {
  
  # read and parse the dvw file
  dv       <- dv_read(dvw_path)
  plays_df <- plays(dv)
  
  # isolate skills we care about, per point
  serves     <- plays_df |> filter(skill == "Serve")
  receptions <- plays_df |> filter(skill == "Reception")
  sets       <- plays_df |> filter(skill == "Set",    phase == "Reception")
  attacks    <- plays_df |> filter(skill == "Attack",  phase == "Reception")
  
  # for each skill, keep only first occurrence per point (should only be 1)
  
  serve_first <- serves |>
    group_by(point_id) |>
    slice(1) |>
    ungroup() |>
    select(
      # match details
      point_id,
      set_number,
      home_team, visiting_team,
      home_team_id, visiting_team_id,
      home_score_start_of_point  = home_team_score,
      visiting_score_start_of_point = visiting_team_score,
      # server info
      server_id       = player_id,
      serve_eval_code = evaluation_code,
      serving_team    = team,
      # position info
      home_setter_position,
      visiting_setter_position,
      home_p1, home_p2, home_p3, home_p4, home_p5, home_p6,
      visiting_p1, visiting_p2, visiting_p3, visiting_p4, visiting_p5, visiting_p6,
      home_player_id1, home_player_id2, home_player_id3,
      home_player_id4, home_player_id5, home_player_id6,
      visiting_player_id1, visiting_player_id2, visiting_player_id3,
      visiting_player_id4, visiting_player_id5, visiting_player_id6
    )
  
  reception_first <- receptions |>
    group_by(point_id) |>
    slice(1) |>
    ungroup() |>
    select(
      point_id,
      receiver_id       = player_id,
      receive_eval_code = evaluation_code,
      receiving_team    = team
    )
  
  set_first <- sets |>
    group_by(point_id) |>
    slice(1) |>
    ungroup() |>
    select(
      point_id,
      setter_id = player_id,
      set_code,
      set_type
    )
  
  attack_first <- attacks |>
    group_by(point_id) |>
    slice(1) |>
    ungroup() |>
    select(
      point_id,
      attacker_id       = player_id,
      attack_code,
      attack_eval_code  = evaluation_code
    )
  
  point_outcome <- plays_df |>
    filter(!is.na(point_won_by)) |>
    group_by(point_id) |>
    slice(1) |>
    ungroup() |>
    select(
      point_id,
      point_won_by
    )
  
  # join together by reception — every serve-receive point must have a reception
  # set and attack use left_join in case absent (ace, overpass, etc.)
  
  result <- serve_first |>
    inner_join(reception_first, by = "point_id") |>
    left_join(set_first,        by = "point_id") |>
    left_join(attack_first,     by = "point_id") |>
    left_join(point_outcome,    by = "point_id")
  

  # --- player name lookup ---
  # extract id -> name mapping from all rows in the file
  player_names <- plays_df |>
    filter(!is.na(player_id), !is.na(player_name)) |>
    select(player_id, player_name) |>
    distinct()

  # join names for each role
  result <- result |>
    left_join(player_names |> rename(server_name   = player_name),
              by = c("server_id"   = "player_id")) |>
    left_join(player_names |> rename(receiver_name = player_name),
              by = c("receiver_id" = "player_id")) |>
    left_join(player_names |> rename(setter_name   = player_name),
              by = c("setter_id"   = "player_id")) |>
    left_join(player_names |> rename(attacker_name = player_name),
              by = c("attacker_id" = "player_id"))
  
  # compute score_diff from the focus team's perspective
  # (positive means focus team is winning at start of point)
  
  result <- result |>
    mutate(
      score_diff = case_when(
        home_team_id == focus_team_id ~
          home_score_start_of_point - visiting_score_start_of_point,
        visiting_team_id == focus_team_id ~
          visiting_score_start_of_point - home_score_start_of_point,
        TRUE ~ NA_real_
      )
    )
  
  # flag special situations
  
  result <- result |>
    mutate(
      is_ace      = serve_eval_code == "#",
      is_overpass = receive_eval_code == "/" | 
        (!is.na(receive_eval_code) & receive_eval_code == "="),
      out_of_system = receive_eval_code %in% c("-", "!"),
      no_set      = is.na(setter_id),
      no_attack   = is.na(attacker_id)
    )
  
  # enforce that receiving_team matches the team actually receiving 
  
  result <- result |>
    filter(serving_team != receiving_team)
  
  # lag feature: did we win the previous serve-receive point? ---
  result <- result |>
    arrange(set_number, point_id) |>
    mutate(
      prev_point_won = lag(point_won_by) == focus_team_name,
      prev_attacker  = lag(attacker_id)   # who was set last time
    )

  # column ordering for readability
  
  result <- result |>
    select(
      point_id, set_number,
      serving_team, receiving_team,
      home_team, visiting_team,
      home_team_id, visiting_team_id,
      home_setter_position, visiting_setter_position,
      home_p1:home_p6,
      visiting_p1:visiting_p6,
      home_player_id1:home_player_id6,
      visiting_player_id1:visiting_player_id6,
      server_id, server_name, serve_eval_code,
      receiver_id, receiver_name, receive_eval_code,
      setter_id, setter_name, set_code, set_type,
      attacker_id, attacker_name, attack_code, attack_eval_code,
      home_score_start_of_point, visiting_score_start_of_point,
      score_diff,
      point_won_by,
      prev_point_won,
      is_ace, is_overpass, out_of_system, no_set, no_attack
    )
  return(result)
}

# run and export (reads arguments from command line) ---
args <- commandArgs(trailingOnly = TRUE)

dvw_path        <- args[1]
focus_team_id   <- as.numeric(args[2])
focus_team_name <- args[3]
output_path     <- args[4]

result <- extract_serve_receive_plays(dvw_path, focus_team_id, focus_team_name)
write_csv(result, output_path)
cat("Export complete:", nrow(result), "rows written to", output_path, "\n")