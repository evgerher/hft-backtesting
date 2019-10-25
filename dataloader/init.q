reload_index:{`index_table set ([] symbol:`symbol$(); timestamp:`timestamp$(); price: `float$())}
reload_snapshot:{`snapshot_table set ([] timestamp:`timestamp$(); symbol:`symbol$(); `float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$();`float$(); `long$())}
reload_index[]
reload_snapshot[]
\g
\p 5002
echo_it:{show index_table}
echo_st:{show snapshot_table}
store_index:{save `:index_table.csv}
store_snapshot:{save `:snapshot_table.csv}
//.u.reload_tables:{reload_index; reload_snapshot \ []}
\l safe_update.q
