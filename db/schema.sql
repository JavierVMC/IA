DROP DATABASE IF EXISTS ia_project;
CREATE DATABASE ia_project;

USE ia_project;

CREATE TABLE registers(
	id int auto_increment,
    fecha datetime not null,
    curso varchar(255) not null,
    paralelo int not null,
    horario varchar(255) not null,
    carrera varchar(255) not null,
    facultad varchar(255) not null,
    PRIMARY KEY (id)
);
