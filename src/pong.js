import Phaser from "phaser";
import ballPath from './vendor/assets/images/ball.png'
import paddlePath from './vendor/assets/images/paddle-horizontal.png'

const isMobile = window.innerWidth < 450
const scaleDownSketch = !isMobile

const config = {
    type: Phaser.AUTO,
    parent: 'main-canvas',
    width: scaleDownSketch ? window.innerWidth / 1.2 : window.innerWidth,
    height: scaleDownSketch ? window.innerHeight / 1.3 : window.innerHeight / 1.2,
    scene: {
        preload,
        create,
        update,
    },
    pixelArt: true,
    physics: {
        default: 'arcade',
        arcade: {
            gravity: false,
        }
    }
};

const game = new Phaser.Game(config);

let player1, player2, ball, cursors;
let gameStarted = false;
let openingText;

function preload() {
    this.load.image('ball', ballPath);
    this.load.image('paddle', paddlePath);
}

function create() {
    ball = this.physics.add.sprite(
        this.physics.world.bounds.width / 2, // x position
        this.physics.world.bounds.height / 2, // y position
        'ball' // key of image for the sprite
    );
    ball.setVisible(false);

    player1 = this.physics.add.sprite(
        this.physics.world.bounds.width / 2, // x position
        this.physics.world.bounds.height, // y position
        'paddle', // key of image for the sprite
    );

    player2 = this.physics.add.sprite(
        this.physics.world.bounds.width / 2, // x position
        0, // y position
        'paddle', // key of image for the sprite
    );

    cursors = this.input.keyboard.createCursorKeys();

    player1.setCollideWorldBounds(true);
    player2.setCollideWorldBounds(true);
    ball.setCollideWorldBounds(true);
    ball.setBounce(1, 1);
    player1.setImmovable(true);
    player2.setImmovable(true);
    this.physics.add.collider(ball, player1, null, null, this);
    this.physics.add.collider(ball, player2, null, null, this);

    openingText = this.add.text(
        this.physics.world.bounds.width / 2,
        this.physics.world.bounds.height / 2,
        'Press SPACE to Start',
        {
            fontFamily: 'Monaco, Courier, monospace',
            fontSize: '50px',
            fill: '#fff'
        }
    );

    openingText.setOrigin(0.5);
}

const paddleSpeed = 400

function update() {
    player1.body.setVelocityX(0);
    player2.body.setVelocityX(0);

    // manage events for neck stretches
    
    if (window.gameLeftMove()) {
        player1.body.setVelocityX(paddleSpeed*-1);
        player2.body.setVelocityX(paddleSpeed*-1);
    } else if (window.gameRightMove()) {
        player1.body.setVelocityX(paddleSpeed);
        player2.body.setVelocityX(paddleSpeed);
    }
    if (!gameStarted) {
        if (cursors.space.isDown) {
            ball.setVisible(true);
            gameStarted = true;
            const initialXSpeed = 250;
            const initialYSpeed = 200;
            ball.setVelocityX(initialXSpeed);
            ball.setVelocityY(initialYSpeed);
            openingText.setVisible(false);
        }
    }
}
