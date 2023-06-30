### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ ebddad20-13a1-11ee-04f3-f9da86d3eb04
begin
	using Pkg
	Pkg.activate(".")
	Pkg.add("DataFrames")
	using CSV
	using DataFrames
end

# ╔═╡ 8ed51274-a08d-4b5d-8ed9-3edf0fc9ce6d
using Dates

# ╔═╡ f5086edf-e5b9-4c58-b96c-a811b5865c63
md"""
# Helper functions.
"""

# ╔═╡ 61d31a03-b4ba-4917-9022-9a9f202a9777
"""
	one_hot_helper(i, total): Helper function for `one_hot`
"""
function one_hot_helper(i, total)
	rslt = falses(total)
	rslt[i] = true
	return rslt
end

# ╔═╡ 1154f824-c1cc-42a3-af69-087793b557bd
"""
	one_hot(col): This function returns a dictionary representing one-hot encoding of the input column.
"""
function one_hot(col)
	uniques = unique(col)
	num_entires, num_uniques = size(col)[1], size(uniques)[1]
	rslt = Array{Bool, 2}(undef, num_entires, num_uniques)
	temp_dict = Dict()
	# build temp_dict
	for (i, curr_value) in enumerate(uniques)
		temp_dict[curr_value] = i
	end
	# build rslt; No data racing will happen so it is safe to use multithreading
	Threads.@threads for i = 1 : num_entires
		rslt[i,:] = one_hot_helper(temp_dict[col[i]], num_uniques)
	end
	return rslt
end

# ╔═╡ ee7700ff-6f55-4c94-be09-b1c1cd05be31
md"""
# 1. Read `realtor-data.csv`
"""

# ╔═╡ 072c77a4-1541-4044-8e7c-e017b17830e2
md"""
## 1.1 Read Data
"""

# ╔═╡ dedbe010-cfa5-43a0-946e-69dcb4b391bd
df = CSV.read("realtor-data.csv", DataFrame)

# ╔═╡ 1dce3426-8e64-426b-a6a2-0004fb766a6e
md"""
## 1.2 Check Data
"""

# ╔═╡ cb58c290-eb20-4a31-b7c6-f4b2b3ac1e8e
begin
	num_entry, num_header = size(df)
	headers = names(df)
	print("Found ")
	printstyled(num_entry; color = :yellow)
	println(" entries.")
	print("Found ")
	printstyled(num_header; color = :yellow)
	println(" headers:")
	for (i,h) in enumerate(headers)
		println("\t$i: $h")
	end
end

# ╔═╡ 201789c1-b2d7-4ea7-b06f-d8b4f8f36be2
md"""
## 1.3 Clean Data
"""

# ╔═╡ c387e0e6-93a6-4052-8b5f-36ad4095aa41
md"""
### 1.3.1 Delete `full_address` and `street`
	Since almost all entries have unique adress and it's hard to apply one-hot encoding on these address. Plus, I believe `city` can present the sense of 'area' for the NN to make predictions.
"""

# ╔═╡ 4c9ea0db-0d54-4bf2-85a7-b15635db0a2f
begin
	# Delete `full_address`
	"full_address" in names(df) && select!(df, Not(:full_address)) 
	# Delete `street`
	"street" in names(df) && select!(df, Not(:street))
end;

# ╔═╡ c1f9be36-c6ff-4322-b82a-db829abb8ad0
df

# ╔═╡ 7495dcf6-f6db-4653-a9e6-c0033ed3d974
md"""
### 1.3.2 Clean `status` 
"""

# ╔═╡ db5abe2c-74e9-4bb4-92f2-f403a57c57b4
curr_unique = unique(df[!, :status])

# ╔═╡ de0035a8-d1e6-48b8-a0f6-d6b998bfda93
md"""
So `status` column looks alreay clean. Applying One-hot encoding.
"""

# ╔═╡ fa8b7b09-227e-4fbd-a9c0-dba5f3d6ada3
begin
	if "status" in names(df)
		curr_name = "status"
		num_uniques = size(curr_unique)[1]
		new_status_col = one_hot(df[!, :status])
		# Add new col to df
		for i = 1 : num_uniques
			df[!, "$(curr_name)_$i"] = new_status_col[:, i]
		end 
		# Delete `status`
		select!(df, Not(:status))
		# Free memory
		new_status_col = nothing
	end
end;

# ╔═╡ fc9f928c-df39-495d-8e01-7ce0239e0d6b
df

# ╔═╡ f4b0a019-ce83-4991-ab0e-7c9758911e7b
md"""
### 1.3.2 Clean `price` 
"""

# ╔═╡ 4a926403-7461-4768-ab5e-a8d3e7548618
sort(unique(df[!, :price]))

# ╔═╡ 188b657d-598f-415a-910d-a8fa362900c4
md"""
	So `price` column looks alreay clean and no need to apply One-hot encoding.
"""

# ╔═╡ ccb517d3-d5ae-4388-bce1-c28feb62ae04
md"""
### 1.3.2 Clean `bed` 
"""

# ╔═╡ 92f0b495-d6c4-4939-aefd-73776e7feb49
sort(unique(df[!, :bed]))

# ╔═╡ 9176c11b-3e54-43e0-907a-5ff322cdcd2c
md"""
Found 'missing' in `price`.
"""

# ╔═╡ cfb329c8-c464-42c8-adcb-9890f177150e
begin
	if "bed" in names(df)
		# Count number of `missing`
		bed_missing_ct = count(i -> (typeof(i)==Missing), df[!, :bed])
		print("Found ")
		printstyled(bed_missing_ct; color = :yellow)
		println(" 'missing'.")
	end
end

# ╔═╡ 0a22cf9a-d60d-4895-8339-16299b419ad9
md"""
Sicne 98937 entries have value = 'missing', so the best way to deal with 'missing' is to apply semi-one-hot.
"""

# ╔═╡ 2da8728e-2da3-443f-8e18-f536c7bb8978
begin
	if "bed" in names(df)
		new_bed_col = Array{Int, 2}(undef, size(df)[1], 2)
		Threads.@threads for i = 1 : size(df)[1]
			if typeof(df[!, :bed][i])==Missing
				new_bed_col[i, :] = [true, 0]
			else
				new_bed_col[i, :] = [false, df[!, :bed][i]]
			end
		end
		# Add new col to df
		df[!, "bed_is_missing"] = new_bed_col[:, 1]
		df[!, "bed_value"] = new_bed_col[:, 2]
		# Delete `bed`
		select!(df, Not(:bed))
		# Free memory
		new_bed_col = nothing
	end
end

# ╔═╡ baa16f18-c1fe-4789-a1a5-0b26b07f4a0f
df

# ╔═╡ eab947d4-6aa5-4723-8d3f-043c4ef0ca5e
md"""
### 1.3.3 Clean `bath` 
"""

# ╔═╡ 543d67e1-7804-4d27-aabd-cf381f926bd6
sort(unique(df[!, :bath]))

# ╔═╡ c61b138d-109d-4e11-9fcc-8f90ce673ef3
md"""
Found 'missing' in `bath`.
"""

# ╔═╡ 2ff7f585-56d5-42d7-8e47-6d0417715692
begin
	if "bath" in names(df)
		# Count number of `missing`
		bath_missing_ct = count(i -> (typeof(i)==Missing), df[!, :bath])
		print("Found ")
		printstyled(bath_missing_ct; color = :yellow)
		println(" 'missing'.")
	end
end

# ╔═╡ 0ea306bb-424f-46a3-9f9b-24b6bef11d9d
md"""
Sicne 95218 entries have value = 'missing', so the best way to deal with 'missing' is to apply semi-one-hot.
"""

# ╔═╡ 244b4cc7-ac75-40d5-af61-ecab9e062aff
begin
	if "bath" in names(df)
		new_bath_col = Array{Int, 2}(undef, size(df)[1], 2)
		Threads.@threads for i = 1 : size(df)[1]
			if typeof(df[!, :bath][i])==Missing
				new_bath_col[i, :] = [true, 0]
			else
				new_bath_col[i, :] = [false, df[!, :bath][i]]
			end
		end
		# Add new col to df
		df[!, "bath_is_missing"] = new_bath_col[:, 1]
		df[!, "bath_value"] = new_bath_col[:, 2]
		# Delete `bath`
		select!(df, Not(:bath))
		# Free memory
		new_bath_col = nothing
	end
end

# ╔═╡ fb264441-494c-47a4-8912-8dda57000b48
df

# ╔═╡ 7ee7f1f3-b408-4464-9933-1f2416d0f427
md"""
### 1.3.4 Clean `acre_lot` 
"""

# ╔═╡ f6b6b1b4-57f3-496a-83b1-d8954a68a437
sort(unique(df[!, :acre_lot]))

# ╔═╡ e688cd69-1b9c-4998-b55a-6806b28695b9
md"""
Found 'missing' in `acre_lot`.
"""

# ╔═╡ 29f99f4f-984f-4914-840e-3b8c4a1dc505
begin
	if "acre_lot" in names(df)
		# Count number of `missing`
		acre_lot_missing_ct = count(i -> (typeof(i)==Missing), df[!, :acre_lot])
		print("Found ")
		printstyled(acre_lot_missing_ct; color = :yellow)
		println(" 'missing'.")
	end
end

# ╔═╡ 7ff51ab5-e990-475a-827d-21596e796665
md"""
Sicne 104979  entries have value = 'missing', so the best way to deal with 'missing' is to apply semi-one-hot.
"""

# ╔═╡ f7401d7b-4f65-423c-a481-7e8d23852af0
begin
	if "acre_lot" in names(df)
		new_acre_lot_col = Array{Float64, 2}(undef, size(df)[1], 2)
		Threads.@threads for i = 1 : size(df)[1]
			if typeof(df[!, :acre_lot][i])==Missing
				new_acre_lot_col[i, :] = [true, 0]
			else
				new_acre_lot_col[i, :] = [false, df[!, :acre_lot][i]]
			end
		end
		# Add new col to df
		df[!, "acre_lot_is_missing"] = new_acre_lot_col[:, 1]
		df[!, "acre_lot_value"] = new_acre_lot_col[:, 2]
		# Delete `acre_lot`
		select!(df, Not(:acre_lot))
		# Free memory
		new_acre_lot_col = nothing
	end
end

# ╔═╡ 36da5666-badf-47af-80bc-f6f2bde5ddee
df

# ╔═╡ e0f68edb-434a-4927-b843-f851c06c7cee
md"""
### 1.3.5 Clean `city` 
"""

# ╔═╡ bb87a850-5221-4cf0-9e9a-87a9ead4bc02
sort(unique(df[!, :city]))

# ╔═╡ 97fd431b-633e-4bec-8afc-f47e1117a8d3
md"""
Found 'missing' in `city`.
"""

# ╔═╡ 03bab8eb-d903-4094-9f3a-6e5fe36996ee
begin
	if "city" in names(df)
		# Count number of `missing`
		city_missing_ct = count(i -> (typeof(i)==Missing), df[!, :city])
		print("Found ")
		printstyled(city_missing_ct; color = :yellow)
		println(" 'missing'.")
	end
end

# ╔═╡ 930c2e90-b629-453f-abda-fe4e9165a4ce
md"""
Sicne only 59 entries have value = 'missing', so the best way to deal with 'missing' is to delete them and apply one-hot encoding.
"""

# ╔═╡ 60a1b5e9-9415-46bd-a7d7-d606b5bd7f7f
begin
	if "city" in names(df)
		# Delete 'missing' entries
		city_rows_to_delte_idx = []
		for (i, curr_value) in enumerate(df[!, :city])
			if typeof(curr_value)==Missing
				push!(city_rows_to_delte_idx, i)
			end
		end
		delete!(df, city_rows_to_delte_idx)
	end
end

# ╔═╡ fa3dbf62-0d0f-4360-8974-14606095c1f4
begin
	# Check if all 'missing' rows got deleted.
	# Count number of `missing`
	city_missing_ct2 = count(i -> (typeof(i)==Missing), df[!, :city])
	print("Found ")
	printstyled(city_missing_ct2; color = :yellow)
	println(" 'missing'.")
end

# ╔═╡ 059a3467-cfcd-4fee-87bb-c4bc5de40f06
md"""
Now, Apply one-hot.
"""

# ╔═╡ 9da02994-75c8-41b3-a8ee-524e5c3cfba2
begin
	if "city" in names(df)
		curr_name2 = "city"
		curr_city_unique = unique(df[!, :city])
		city_num_uniques = size(curr_city_unique)[1]
		new_city_col = one_hot(df[!, :city])
		# Add new col to df
		for i = 1 : city_num_uniques
			df[!, "$(curr_name2)_$i"] = new_city_col[:, i]
		end 
		# Delete `city`
		select!(df, Not(:city))
		# Free memory
		new_city_col = nothing
	end
end

# ╔═╡ bf2b74fa-5faf-4177-9df9-4be00f5cd048
df

# ╔═╡ fea47df4-586b-4145-84c3-5984db79065e
md"""
### 1.3.5 Clean `state` 
"""

# ╔═╡ f5571227-bc61-455f-bb02-3b5962088af3
sort(unique(df[!, :state]))

# ╔═╡ 1fc61fdb-a3d5-4c58-85fa-03a4f2a2bbb8
md"""
'state' looks clean. Apply one-hot.
"""

# ╔═╡ d94bfea5-3227-48a6-b815-567e7ea33afd
begin
	if "state" in names(df)
		curr_name3 = "state"
		curr_state_unique = unique(df[!, :state])
		state_num_uniques = size(curr_state_unique)[1]
		new_state_col = one_hot(df[!, :state])
		# Add new col to df
		for i = 1 : state_num_uniques
			df[!, "$(curr_name3)_$i"] = new_state_col[:, i]
		end 
		# Delete `state`
		select!(df, Not(:state))
		# Free memory
		new_state_col = nothing
	end
end

# ╔═╡ a60d57a8-3bf0-4a10-af7e-4852b8428918
df

# ╔═╡ 47a645fa-974c-43b5-abeb-3dac9e8b260f
names(df)

# ╔═╡ 722ecdaa-c9ef-496d-8de2-7aafb9f548a8
md"""
### 1.3.6 Clean `zip_code` 
"""

# ╔═╡ a96def54-d135-48f9-9695-2d086267c44e
sort(unique(df[!, :zip_code]))

# ╔═╡ da7d64ad-127f-4999-87ae-ca1a79bfb046
md"""
Found 'missing' in `zip_code`.
"""

# ╔═╡ 199da9c5-8c73-44e8-b87a-ef1f6c629b26
begin
	if "zip_code" in names(df)
		# Count number of `missing`
		zip_code_missing_ct = count(i -> (typeof(i)==Missing), df[!, :zip_code])
		print("Found ")
		printstyled(zip_code_missing_ct; color = :yellow)
		println(" 'missing'.")
	end
end

# ╔═╡ ff57b1fe-822d-4251-bf68-5c1b7af9c8bf
md"""
Sicne only 146  entries have value = 'missing', so the best way to deal with 'missing' is to delete them and apply one-hot encoding. Zip codes are numbers, however, they don't have continues meanigns. So still need to apply one-hot.
"""

# ╔═╡ 16a2855f-915f-4893-aa7a-b36c8267b8d0
begin
	if "zip_code" in names(df)
		# Delete 'missing' entries
		zip_code_rows_to_delte_idx = []
		for (i, curr_value) in enumerate(df[!, :zip_code])
			if typeof(curr_value)==Missing
				push!(zip_code_rows_to_delte_idx, i)
			end
		end
		delete!(df, zip_code_rows_to_delte_idx)
	end
end

# ╔═╡ 9e04544e-534a-4ff3-98ba-f4594b6b9d30
begin
	# Check if all 'missing' rows got deleted.
	# Count number of `missing`
	zip_code_missing_ct2 = count(i -> (typeof(i)==Missing), df[!, :zip_code])
	print("Found ")
	printstyled(zip_code_missing_ct2; color = :yellow)
	println(" 'missing'.")
end

# ╔═╡ 2b118dd3-d6cc-461d-81a7-568b56b66a14
md"""
Conver numbers to String to speed up.
"""

# ╔═╡ d045f347-f26e-42dd-8d4e-fa6895dd54cc
begin
	zip_code_ct = size(df[!, :zip_code])[1]
	zip_code_col_String = Array{String, 1}(undef, zip_code_ct)
	Threads.@threads for i = 1 : zip_code_ct
		zip_code_col_String[i] = string(round(Int, df[!, :zip_code][i]))
	end
end

# ╔═╡ defd94d1-e912-4468-9d42-2d6a0ba50c31
md"""
Now, Apply one-hot.
"""

# ╔═╡ 0f4e5aa4-baf4-494b-8775-f79eb48e2543
begin
	curr_name4 = "zip_code"
	curr_zip_code_unique = unique(zip_code_col_String)
	zip_code_num_uniques = size(curr_zip_code_unique)[1]
	new_zip_code_col = one_hot(zip_code_col_String)
	# Add new col to df
	for i = 1 : zip_code_num_uniques
		df[!, "$(curr_name4)_$i"] = new_zip_code_col[:, i]
	end 
	# Delete `zip_code`
	select!(df, Not(:zip_code))
	# Free memory
	new_zip_code_col = nothing
end

# ╔═╡ 159a6c77-d679-4902-92b9-e8065a7a3ad1
df

# ╔═╡ be46f584-e703-4673-830b-05fec285948b
md"""
### 1.3.7 Clean `house_size` 
"""

# ╔═╡ 209be7e2-4eb7-4fb6-b432-1fee19f7574e
sort(unique(df[!, :house_size]))

# ╔═╡ 20ffd9d8-04fd-4ec1-a9f0-7ef81588d0da
md"""
Found 'missing' in `house_size`.
"""

# ╔═╡ 5d6bc3a1-0885-4d39-8815-7bfb2547515a
begin
	if "house_size" in names(df)
		# Count number of `missing`
		house_size_missing_ct = count(i -> (typeof(i)==Missing), df[!, :house_size])
		print("Found ")
		printstyled(house_size_missing_ct; color = :yellow)
		println(" 'missing'.")
	end
end

# ╔═╡ 94bd61e5-39dd-4f9c-939a-dfaf94837615
md"""
Sicne 116384  entries have value = 'missing', so the best way to deal with 'missing' is to apply semi-one-hot.
"""

# ╔═╡ 95aeb356-957f-461b-92ca-b485ed208337
begin
	if "house_size" in names(df)
		new_house_size_col = Array{Float64, 2}(undef, size(df)[1], 2)
		Threads.@threads for i = 1 : size(df)[1]
			if typeof(df[!, :house_size][i])==Missing
				new_house_size_col[i, :] = [true, 0]
			else
				new_house_size_col[i, :] = [false, df[!, :house_size][i]]
			end
		end
		# Add new col to df
		df[!, "house_size_is_missing"] = new_house_size_col[:, 1]
		df[!, "house_size_value"] = new_house_size_col[:, 2]
		# Delete `house_size`
		select!(df, Not(:house_size))
		# Free memory
		new_house_size_col = nothing
	end
end

# ╔═╡ e8c9d6e0-8ba9-4b20-887e-8c57bfac447f
df

# ╔═╡ 74950001-758f-4345-b261-6c32f2d78f5a
names(df)

# ╔═╡ defddda2-513e-4026-b005-354b43cbf81d
md"""
### 1.3.8 Clean `sold_date` 
"""

# ╔═╡ fd9d8e0c-d47a-4b21-85ba-b464f04eee0a
sort(unique(df[!, :sold_date]))

# ╔═╡ d97cbbaa-95b3-4add-9557-2fff19f4d4e0
md"""
Found 'missing' in `sold_date`.
"""

# ╔═╡ f1511fdc-7570-4bdf-8cfb-3a6f2fdcc97d
begin
	if "sold_date" in names(df)
		# Count number of `missing`
		sold_date_missing_ct = count(i -> (typeof(i)==Missing), df[!, :sold_date])
		print("Found ")
		printstyled(sold_date_missing_ct; color = :yellow)
		println(" 'missing'.")
	end
end

# ╔═╡ 4d9f36b9-8406-4bed-9da6-72b69174f927
md"""
Sicne 309447 entries have value = 'missing', so the best way to deal with 'missing' is to apply semi-one-hot. Since there are 8220 unique date, one-hot all of them will result in too many features. So here I'm keeping only "Year" not month or day.
"""

# ╔═╡ 3b964fd6-239c-4c6f-ad8e-7ee92dd97ec8
begin
	# Convert 'YYYY-MM-DD' to `YYYY`
	new_date_col_String = Array{String, 1}(undef, size(df)[1])
	Threads.@threads for i = 1 : size(df)[1]
		if typeof(df[!, :sold_date][i])!=Missing
			new_date_col_String[i] = Dates.format(df[!, :sold_date][i], "yyyy")
		else
			new_date_col_String[i] = "0000"
		end
	end
end

# ╔═╡ 9c5d211c-de66-4e0e-a541-3480512b0e78
begin
	curr_name5 = "sold_date"
	curr_sold_date_unique = unique(new_date_col_String)
	sold_date_num_uniques = size(curr_sold_date_unique)[1]
	new_sold_date_col = one_hot(new_date_col_String)
	# Add new col to df
	for i = 1 : sold_date_num_uniques
		df[!, "$(curr_name5)_$i"] = new_sold_date_col[:, i]
	end 
	# Delete `sold_date`
	select!(df, Not(:sold_date))
	# Free memory
	new_sold_date_col = nothing
end

# ╔═╡ 2496c417-a36f-4a51-b868-d6633cf8c202
df

# ╔═╡ 760105e2-ff02-4627-8a7b-1f98ff29dbea
md"""
# 2. Save as new `csv`
"""

# ╔═╡ c400fa65-9875-4f53-995a-710b99afaab3
CSV.write(raw"C:\Users\wenbl13\OneDrive - UCI Health\Desktop\simple ML\clean_data.csv", df) 

# ╔═╡ Cell order:
# ╠═ebddad20-13a1-11ee-04f3-f9da86d3eb04
# ╟─f5086edf-e5b9-4c58-b96c-a811b5865c63
# ╠═1154f824-c1cc-42a3-af69-087793b557bd
# ╠═61d31a03-b4ba-4917-9022-9a9f202a9777
# ╟─ee7700ff-6f55-4c94-be09-b1c1cd05be31
# ╟─072c77a4-1541-4044-8e7c-e017b17830e2
# ╠═dedbe010-cfa5-43a0-946e-69dcb4b391bd
# ╟─1dce3426-8e64-426b-a6a2-0004fb766a6e
# ╠═cb58c290-eb20-4a31-b7c6-f4b2b3ac1e8e
# ╟─201789c1-b2d7-4ea7-b06f-d8b4f8f36be2
# ╟─c387e0e6-93a6-4052-8b5f-36ad4095aa41
# ╠═4c9ea0db-0d54-4bf2-85a7-b15635db0a2f
# ╠═c1f9be36-c6ff-4322-b82a-db829abb8ad0
# ╟─7495dcf6-f6db-4653-a9e6-c0033ed3d974
# ╠═db5abe2c-74e9-4bb4-92f2-f403a57c57b4
# ╟─de0035a8-d1e6-48b8-a0f6-d6b998bfda93
# ╠═fa8b7b09-227e-4fbd-a9c0-dba5f3d6ada3
# ╠═fc9f928c-df39-495d-8e01-7ce0239e0d6b
# ╟─f4b0a019-ce83-4991-ab0e-7c9758911e7b
# ╠═4a926403-7461-4768-ab5e-a8d3e7548618
# ╟─188b657d-598f-415a-910d-a8fa362900c4
# ╟─ccb517d3-d5ae-4388-bce1-c28feb62ae04
# ╠═92f0b495-d6c4-4939-aefd-73776e7feb49
# ╟─9176c11b-3e54-43e0-907a-5ff322cdcd2c
# ╠═cfb329c8-c464-42c8-adcb-9890f177150e
# ╟─0a22cf9a-d60d-4895-8339-16299b419ad9
# ╠═2da8728e-2da3-443f-8e18-f536c7bb8978
# ╠═baa16f18-c1fe-4789-a1a5-0b26b07f4a0f
# ╟─eab947d4-6aa5-4723-8d3f-043c4ef0ca5e
# ╠═543d67e1-7804-4d27-aabd-cf381f926bd6
# ╟─c61b138d-109d-4e11-9fcc-8f90ce673ef3
# ╠═2ff7f585-56d5-42d7-8e47-6d0417715692
# ╟─0ea306bb-424f-46a3-9f9b-24b6bef11d9d
# ╠═244b4cc7-ac75-40d5-af61-ecab9e062aff
# ╠═fb264441-494c-47a4-8912-8dda57000b48
# ╟─7ee7f1f3-b408-4464-9933-1f2416d0f427
# ╠═f6b6b1b4-57f3-496a-83b1-d8954a68a437
# ╟─e688cd69-1b9c-4998-b55a-6806b28695b9
# ╠═29f99f4f-984f-4914-840e-3b8c4a1dc505
# ╟─7ff51ab5-e990-475a-827d-21596e796665
# ╠═f7401d7b-4f65-423c-a481-7e8d23852af0
# ╠═36da5666-badf-47af-80bc-f6f2bde5ddee
# ╟─e0f68edb-434a-4927-b843-f851c06c7cee
# ╟─bb87a850-5221-4cf0-9e9a-87a9ead4bc02
# ╟─97fd431b-633e-4bec-8afc-f47e1117a8d3
# ╠═03bab8eb-d903-4094-9f3a-6e5fe36996ee
# ╟─930c2e90-b629-453f-abda-fe4e9165a4ce
# ╠═60a1b5e9-9415-46bd-a7d7-d606b5bd7f7f
# ╠═fa3dbf62-0d0f-4360-8974-14606095c1f4
# ╟─059a3467-cfcd-4fee-87bb-c4bc5de40f06
# ╠═9da02994-75c8-41b3-a8ee-524e5c3cfba2
# ╠═bf2b74fa-5faf-4177-9df9-4be00f5cd048
# ╟─fea47df4-586b-4145-84c3-5984db79065e
# ╠═f5571227-bc61-455f-bb02-3b5962088af3
# ╟─1fc61fdb-a3d5-4c58-85fa-03a4f2a2bbb8
# ╠═d94bfea5-3227-48a6-b815-567e7ea33afd
# ╠═a60d57a8-3bf0-4a10-af7e-4852b8428918
# ╠═47a645fa-974c-43b5-abeb-3dac9e8b260f
# ╟─722ecdaa-c9ef-496d-8de2-7aafb9f548a8
# ╠═a96def54-d135-48f9-9695-2d086267c44e
# ╟─da7d64ad-127f-4999-87ae-ca1a79bfb046
# ╠═199da9c5-8c73-44e8-b87a-ef1f6c629b26
# ╟─ff57b1fe-822d-4251-bf68-5c1b7af9c8bf
# ╠═16a2855f-915f-4893-aa7a-b36c8267b8d0
# ╠═9e04544e-534a-4ff3-98ba-f4594b6b9d30
# ╟─2b118dd3-d6cc-461d-81a7-568b56b66a14
# ╠═d045f347-f26e-42dd-8d4e-fa6895dd54cc
# ╟─defd94d1-e912-4468-9d42-2d6a0ba50c31
# ╠═0f4e5aa4-baf4-494b-8775-f79eb48e2543
# ╠═159a6c77-d679-4902-92b9-e8065a7a3ad1
# ╟─be46f584-e703-4673-830b-05fec285948b
# ╠═209be7e2-4eb7-4fb6-b432-1fee19f7574e
# ╟─20ffd9d8-04fd-4ec1-a9f0-7ef81588d0da
# ╠═5d6bc3a1-0885-4d39-8815-7bfb2547515a
# ╟─94bd61e5-39dd-4f9c-939a-dfaf94837615
# ╠═95aeb356-957f-461b-92ca-b485ed208337
# ╠═e8c9d6e0-8ba9-4b20-887e-8c57bfac447f
# ╠═74950001-758f-4345-b261-6c32f2d78f5a
# ╟─defddda2-513e-4026-b005-354b43cbf81d
# ╠═8ed51274-a08d-4b5d-8ed9-3edf0fc9ce6d
# ╠═fd9d8e0c-d47a-4b21-85ba-b464f04eee0a
# ╟─d97cbbaa-95b3-4add-9557-2fff19f4d4e0
# ╠═f1511fdc-7570-4bdf-8cfb-3a6f2fdcc97d
# ╟─4d9f36b9-8406-4bed-9da6-72b69174f927
# ╠═3b964fd6-239c-4c6f-ad8e-7ee92dd97ec8
# ╠═9c5d211c-de66-4e0e-a541-3480512b0e78
# ╠═2496c417-a36f-4a51-b868-d6633cf8c202
# ╟─760105e2-ff02-4627-8a7b-1f98ff29dbea
# ╠═c400fa65-9875-4f53-995a-710b99afaab3
