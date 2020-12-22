graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "cast_info"
    columns "note"
    columns "id"
    columns "person_id"
    columns "movie_id"
    columns "person_role_id"
    columns "nr_order"
    columns "role_id"
    columns "__search_id"
    types "string"
    types "number"
    types "number"
    types "number"
    types "number"
    types "number"
    types "number"
    types "number"
    pkey "id"
  ]
  node [
    id 1
    label "char_name"
    columns "name"
    columns "imdb_index"
    columns "name_pcode_nf"
    columns "surname_pcode"
    columns "id"
    columns "imdb_id"
    columns "__search_id"
    types "string"
    types "string"
    types "string"
    types "string"
    types "number"
    types "number"
    types "number"
    pkey "id"
  ]
  node [
    id 2
    label "movie_info"
    columns "info"
    columns "note"
    columns "id"
    columns "movie_id"
    columns "info_type_id"
    columns "__search_id"
    types "string"
    types "string"
    types "number"
    types "number"
    types "number"
    types "number"
    pkey "id"
  ]
  node [
    id 3
    label "name"
    columns "name"
    columns "imdb_index"
    columns "name_pcode_cf"
    columns "name_pcode_nf"
    columns "surname_pcode"
    columns "id"
    columns "imdb_id"
    columns "__search_id"
    types "string"
    types "string"
    types "string"
    types "string"
    types "string"
    types "number"
    types "number"
    types "number"
    pkey "id"
  ]
  node [
    id 4
    label "role_type"
    columns "role"
    columns "id"
    columns "__search_id"
    types "string"
    types "number"
    types "number"
    pkey "id"
  ]
  node [
    id 5
    label "title"
    columns "title"
    columns "imdb_index"
    columns "phonetic_code"
    columns "series_years"
    columns "id"
    columns "kind_id"
    columns "production_year"
    columns "imdb_id"
    columns "episode_of_id"
    columns "season_nr"
    columns "episode_nr"
    columns "__search_id"
    types "string"
    types "string"
    types "string"
    types "string"
    types "number"
    types "number"
    types "number"
    types "number"
    types "number"
    types "number"
    types "number"
    types "number"
    pkey "id"
  ]
  edge [
    source 0
    target 5
    key 0
    col1 "movie_id"
    col2 "id"
    name "-"
  ]
  edge [
    source 0
    target 5
    key 1
    col1 "movie_id"
    col2 "id"
    name "-"
  ]
  edge [
    source 0
    target 5
    key 2
    col1 "movie_id"
    col2 "id"
    name "-"
  ]
  edge [
    source 0
    target 5
    key 3
    col1 "movie_id"
    col2 "id"
    name "-"
  ]
  edge [
    source 0
    target 3
    key 0
    col1 "person_id"
    col2 "id"
    name "-"
  ]
  edge [
    source 0
    target 1
    key 0
    col1 "person_role_id"
    col2 "id"
    name "-"
  ]
  edge [
    source 0
    target 4
    key 0
    col1 "role_id"
    col2 "id"
    name "-"
  ]
  edge [
    source 2
    target 5
    key 0
    col1 "movie_id"
    col2 "id"
    name "-"
  ]
  edge [
    source 2
    target 5
    key 1
    col1 "movie_id"
    col2 "id"
    name "-"
  ]
  edge [
    source 2
    target 5
    key 2
    col1 "movie_id"
    col2 "id"
    name "-"
  ]
  edge [
    source 2
    target 5
    key 3
    col1 "movie_id"
    col2 "id"
    name "-"
  ]
]
