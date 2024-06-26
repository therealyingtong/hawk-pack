db_dev:
	docker run --name hawk-db \
		-p 5432:5432 \
		-e POSTGRES_PASSWORD=x \
		-d postgres

db_ui:
	docker run --name hawk-db-ui \
		-p 5050:80 \
		-e 'PGADMIN_DEFAULT_EMAIL=postgres@inversed.tech' \
		-e 'PGADMIN_DEFAULT_PASSWORD=x' \
		-d dpage/pgadmin4
