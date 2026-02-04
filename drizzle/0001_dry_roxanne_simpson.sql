CREATE TABLE `analyses` (
	`id` int AUTO_INCREMENT NOT NULL,
	`videoId` int NOT NULL,
	`userId` int NOT NULL,
	`mode` enum('all','radar','team','track','players','ball','pitch') NOT NULL,
	`status` enum('pending','uploading','processing','completed','failed') NOT NULL DEFAULT 'pending',
	`progress` int NOT NULL DEFAULT 0,
	`currentStage` varchar(128),
	`errorMessage` text,
	`annotatedVideoUrl` text,
	`radarVideoUrl` text,
	`trackingDataUrl` text,
	`analyticsDataUrl` text,
	`startedAt` timestamp,
	`completedAt` timestamp,
	`processingTimeMs` int,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `analyses_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `commentary` (
	`id` int AUTO_INCREMENT NOT NULL,
	`analysisId` int NOT NULL,
	`eventId` int,
	`frameStart` int,
	`frameEnd` int,
	`type` varchar(64) NOT NULL,
	`content` text NOT NULL,
	`confidence` float,
	`groundingData` json,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `commentary_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `events` (
	`id` int AUTO_INCREMENT NOT NULL,
	`analysisId` int NOT NULL,
	`type` varchar(64) NOT NULL,
	`frameNumber` int NOT NULL,
	`timestamp` float NOT NULL,
	`playerId` int,
	`teamId` int,
	`targetPlayerId` int,
	`startX` float,
	`startY` float,
	`endX` float,
	`endY` float,
	`success` boolean,
	`confidence` float,
	`metadata` json,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `events_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `statistics` (
	`id` int AUTO_INCREMENT NOT NULL,
	`analysisId` int NOT NULL,
	`possessionTeam1` float,
	`possessionTeam2` float,
	`passesTeam1` int,
	`passesTeam2` int,
	`passAccuracyTeam1` float,
	`passAccuracyTeam2` float,
	`shotsTeam1` int,
	`shotsTeam2` int,
	`distanceCoveredTeam1` float,
	`distanceCoveredTeam2` float,
	`avgSpeedTeam1` float,
	`avgSpeedTeam2` float,
	`heatmapDataTeam1` json,
	`heatmapDataTeam2` json,
	`passNetworkTeam1` json,
	`passNetworkTeam2` json,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `statistics_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `tracks` (
	`id` int AUTO_INCREMENT NOT NULL,
	`analysisId` int NOT NULL,
	`frameNumber` int NOT NULL,
	`timestamp` float NOT NULL,
	`playerPositions` json,
	`ballPosition` json,
	`teamFormations` json,
	`voronoiData` json,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `tracks_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `videos` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`title` varchar(255) NOT NULL,
	`description` text,
	`originalUrl` text NOT NULL,
	`fileKey` varchar(512) NOT NULL,
	`duration` float,
	`fps` float,
	`width` int,
	`height` int,
	`frameCount` int,
	`fileSize` int,
	`mimeType` varchar(64),
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `videos_id` PRIMARY KEY(`id`)
);
