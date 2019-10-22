index_table:([] symbol:`symbol$(); timestamp:`timestamp$(); price: `float$())
insert_index:{`index_table insert (x; y; z)}
echo:{show `index_table}